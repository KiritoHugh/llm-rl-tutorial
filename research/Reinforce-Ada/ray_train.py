    def _generate_multi_round_adaptive_downsampling(
        self,
        orig_prompt_batch: DataProto,
        positive_threshold: float = 0.7,
        max_rounds: int = 4,
        round_repeat: int = 8,
        final_keep_per_prompt: int = 4,
        timing_raw: dict | None = None,
        context_batch: DataProto | None = None,
    ):
        """
        Iterative multi-round generation with early stopping downsampling.

        Args:
            orig_prompt_batch: Original prompt batch to generate from
            positive_threshold: Threshold for classifying samples as positive (reward > threshold)
            max_rounds: Maximum number of rounds to perform
            round_repeat: Number of samples to generate per active prompt in each round
            final_keep_per_prompt: Final number of samples to keep per prompt (target: half positive, half negative)
            timing_raw: Optional dict to record timing information
            context_batch: Optional context batch for field alignment via uid

        Returns:
            Tuple of (final_batch, rounds_info) where:
                - final_batch: DataProto with selected samples and aligned context fields
                - rounds_info: Dict with per-round statistics
        """
        # gpt: 该函数用于执行“多轮自适应下采样”生成过程。
        # gpt: 每一轮生成多个候选样本，根据 reward 值筛选出正样本和负样本，
        # gpt: 若某个 prompt 已收集到足够多的正负样本，则提前终止该 prompt 的采样。
        # gpt: 最终每个 prompt 只保留有限数量（final_keep_per_prompt）的样本，以减少训练负担并保证样本质量。

        # Build uid -> fields mapping from context
        ctx_uid_to_fields = {}
        if context_batch is not None:
            ctx_uid_to_fields = build_uid_to_fields_mapping(context_batch)
        # gpt: 建立一个 uid 到上下文字段的映射，用于后续对齐（如奖励计算时保持 prompt 与 context 一致）。

        # Ensure orig_prompt_batch has uid
        ensure_uid_in_batch(orig_prompt_batch, context_batch)
        uid_arr = list(orig_prompt_batch.non_tensor_batch["uid"])
        # gpt: 确保每个样本都有唯一标识 uid，便于跨轮追踪样本状态。

        # Initialize state tracking for each uid
        state = {uid: {"finished": False, "seen": 0, "pos": 0, "neg": 0} for uid in uid_arr}
        # gpt: 初始化每个 uid 的状态信息：
        # gpt: finished -> 是否完成采样
        # gpt: seen -> 已生成的样本数
        # gpt: pos/neg -> 当前累计的正/负样本数量

        # Caches for positive and negative samples per uid
        pos_cache = defaultdict(list)
        neg_cache = defaultdict(list)
        selected_pool_batches: list[DataProto] = []
        selected_count_by_uid = defaultdict(int)
        # For GRPO with global statistics estimation
        uid_full_stats = {uid: {"total_pos": 0, "total_neg": 0} for uid in uid_arr}
        rounds_info = {"per_round": []}
        # gpt: pos_cache/neg_cache 用于暂存每轮采样的正/负样本；
        # gpt: rounds_info 记录每轮统计信息（时间、平均reward、完成数量等）。

        # Main generation loop
        active_uids = set(uid_arr)
        # gpt: active_uids 表示当前仍需要继续采样的 prompt。
        for r in range(max_rounds):
            # gpt: 外层循环控制最多执行 max_rounds 轮采样。
            t0 = time.time()
            if not active_uids:
                # gpt: 若所有 prompt 均已完成，则直接结束循环。
                rounds_info["per_round"].append(
                    {
                        "round": r,
                        "active_prompts": 0,
                        "completed": 0,
                        "finished_prompts": sum(1 for s in state.values() if s["finished"]),
                        "sec": 0.0,
                    }
                )
                break

            # Create mini-batch for active prompts only
            uid_to_idx = {uid: i for i, uid in enumerate(uid_arr)}
            active_indices = [uid_to_idx[uid] for uid in uid_arr if uid in active_uids]
            mini_prompt_batch = orig_prompt_batch[active_indices]
            round_inp = mini_prompt_batch.repeat(repeat_times=round_repeat, interleave=True)
            # gpt: 仅为“活跃的” prompt 创建子批次 mini_prompt_batch。
            # gpt: 然后重复采样 round_repeat 次以生成多个候选输出。

            # Pad to be divisible by dp_size
            dp_size = self.actor_rollout_wg.dp_size if hasattr(self.actor_rollout_wg, "dp_size") else 8
            batch_size = len(round_inp)
            padding_applied = False
            if batch_size % dp_size != 0:
                # gpt: 为了在分布式训练中平衡数据，需要补齐 batch 使其能被 dp_size 整除。
                padding_needed = dp_size - (batch_size % dp_size)
                print(
                    f"Padding batch from {batch_size} to {batch_size + padding_needed} "
                    f"to make it divisible by {dp_size}"
                )
                indices_to_repeat = list(range(batch_size - padding_needed, batch_size))
                if len(indices_to_repeat) == 0:
                    indices_to_repeat = [batch_size - 1] * padding_needed
                padding_batch = round_inp[indices_to_repeat]
                round_inp = DataProto.concat([round_inp, padding_batch])
                padding_applied = True

            # Generate sequences
            gen_out = (
                self.actor_rollout_wg.generate_sequences(round_inp)
                if not self.async_rollout_mode
                else self.async_rollout_manager.generate_sequences(round_inp)
            )
            # gpt: 根据 actor 模型生成文本序列，如果开启异步 rollout，则使用 async manager。

            # Remove padding if applied
            if padding_applied:
                gen_out = gen_out[:batch_size]
                round_inp = round_inp[:batch_size]
            # gpt: 如果前面做过 padding，则在生成完毕后移除补充的样本。

            # Compute rewards for this round
            mini_with_out, seq_reward, uids_round = compute_seq_rewards_for_round(
                mini_prompt_batch=mini_prompt_batch,
                gen_out=gen_out,
                ctx_uid_to_fields=ctx_uid_to_fields,
                reward_fn=self.reward_fn,
                use_rm=self.use_rm,
                rm_wg=self.rm_wg,
                config=self.config,
                kl_ctrl_in_reward=self.kl_ctrl_in_reward if self.config.algorithm.use_kl_in_reward else None,
            )
            seq_reward_np = seq_reward.detach().cpu().numpy().tolist()
            # gpt: 对当前生成的输出计算奖励值（reward），
            # gpt: 支持使用 reward_fn 或 reward model（RM），并在需要时应用 KL 惩罚。

            # Group by uid
            per_uid_local_idx = defaultdict(list)
            for j, uid in enumerate(uids_round):
                per_uid_local_idx[uid].append(j)
            # gpt: 将同一个 prompt（uid）的生成结果聚合到一起，以便单独统计正负样本数量。

            # Update state and cache samples
            completed_this_round = 0
            for uid in list(active_uids):
                locs = per_uid_local_idx.get(uid, [])
                if not locs:
                    continue
                st = state[uid]

                # Cache positive and negative samples
                for j in locs:
                    if st["finished"]:
                        break
                    st["seen"] += 1
                    is_positive = seq_reward_np[j] > positive_threshold
                    if is_positive:
                        st["pos"] += 1
                        pos_cache[uid].append(mini_with_out[[j]])
                        uid_full_stats[uid]["total_pos"] += 1
                    else:
                        st["neg"] += 1
                        neg_cache[uid].append(mini_with_out[[j]])
                        uid_full_stats[uid]["total_neg"] += 1
                # gpt: 将本轮生成结果按 reward 分为正样本和负样本，分别存入缓存。

                def downsample_cache(pos_cache, neg_cache, uid, target_total):
                    # gpt: 下采样函数，根据目标数量从缓存中挑选平衡的正负样本。
                    target_pos = min(target_total // 2, len(pos_cache[uid]))
                    target_neg = min(target_total - target_pos, len(neg_cache[uid]))
                    if target_pos + target_neg < target_total:
                        # gpt: 若不足目标数，则动态补充样本。
                        if len(pos_cache[uid]) > target_pos:
                            additional_pos = min(len(pos_cache[uid]) - target_pos, target_total - (target_pos + target_neg))
                            target_pos += additional_pos
                        elif len(neg_cache[uid]) > target_neg:
                            additional_neg = min(len(neg_cache[uid]) - target_neg, target_total - (target_pos + target_neg))
                            target_neg += additional_neg
                    pos_frags = pos_cache[uid][:target_pos]
                    neg_frags = neg_cache[uid][:target_neg]
                    merged = concat_dataproto_fragments(pos_frags + neg_frags)
                    return merged

                # Check if we have enough samples to finish this uid
                if not st["finished"]:
                    if self.config.algorithm.reinforce_ada_choice == "balanced":
                        # gpt: 模式一：平衡采样，正负样本各占一半。
                        target_pos = final_keep_per_prompt // 2
                        target_neg = final_keep_per_prompt - target_pos

                        if len(pos_cache[uid]) >= target_pos and len(neg_cache[uid]) >= target_neg:
                            merged = downsample_cache(pos_cache, neg_cache, uid, final_keep_per_prompt)
                            selected_pool_batches.append(merged)
                            selected_count_by_uid[uid] = get_first_dim_size(merged)
                            st["finished"] = True
                            completed_this_round += 1

                    else:  # positive_focused
                        # gpt: 模式二：以正样本为主，只要出现足够数量的正样本即可停止。
                        assert self.config.algorithm.reinforce_ada_choice == "positive_focused", (
                            "reinforce_ada_choice has to be one of {'balanced', 'positive_focused'}"
                        )
                        target_pos = 1
                        if len(pos_cache[uid]) >= target_pos:
                            merged = downsample_cache(pos_cache, neg_cache, uid, final_keep_per_prompt)
                            selected_pool_batches.append(merged)
                            selected_count_by_uid[uid] = get_first_dim_size(merged)
                            st["finished"] = True
                            completed_this_round += 1

            # Update active set
            active_uids = {u for u in active_uids if not state[u]["finished"]}
            # gpt: 更新当前活跃的 uid 集合（移除已完成的 prompt）。

            # Record timing and stats
            sec = time.time() - t0
            if timing_raw is not None:
                timing_raw[f"gen_round_{r}_sec"] = sec
            # gpt: 记录本轮耗时，用于性能分析。

            rounds_info["per_round"].append(
                {
                    "round": r,
                    "active_prompts": len(per_uid_local_idx),
                    "completed": completed_this_round,
                    "finished_prompts": sum(1 for s in state.values() if s["finished"]),
                    "reward_mean": float(np.mean(seq_reward_np)) if seq_reward_np else 0.0,
                    "sec": round(sec, 3),
                }
            )
            print(
                f"[Gen-Round {r}] active_prompts={len(per_uid_local_idx)} "
                f"completed={completed_this_round} "
                f"finished={rounds_info['per_round'][-1]['finished_prompts']} "
                f"time={sec:.3f}s "
                f"reward_mean={rounds_info['per_round'][-1]['reward_mean']:.4f}"
            )

            if not active_uids:
                break

        # Handle fallback for uids that didn't reach target
        # gpt: 对于未完成采样的 prompt，执行补救策略（fallback），从已有样本中尽量选出部分作为最终结果。
        uids_that_need_fallback = {uid for uid in uid_arr if not state[uid]["finished"]}

        for uid in uids_that_need_fallback:
            if uid in pos_cache or uid in neg_cache:
                pos_num = len(pos_cache[uid])
                neg_num = len(neg_cache[uid])
                n_rows = pos_num + neg_num
                take = min(final_keep_per_prompt, n_rows)

                if n_rows < final_keep_per_prompt:
                    print(
                        f"[WARN] uid={uid} has {n_rows} samples, less than target "
                        f"{final_keep_per_prompt}, but continuing"
                    )

                if self.config.algorithm.reinforce_ada_choice == "positive_focused":
                    # gpt: 在正样本优先模式下，根据正负比例重新计算应保留数量。
                    ratio = (pos_num / n_rows) if n_rows > 0 else 0.0
                    target_pos = math.ceil(ratio * final_keep_per_prompt)
                    target_pos = max(min(target_pos, take - 1), 1)
                    target_neg = take - target_pos

                actual_pos = min(pos_num, target_pos)
                actual_neg = min(neg_num, target_neg)

                # If one type is insufficient, fill with the other
                if actual_pos + actual_neg < take:
                    if pos_num > actual_pos:
                        additional_pos = min(pos_num - actual_pos, take - actual_pos - actual_neg)
                        actual_pos += additional_pos
                    elif neg_num > actual_neg:
                        additional_neg = min(neg_num - actual_neg, take - actual_pos - actual_neg)
                        actual_neg += additional_neg

                keep_pos = actual_pos
                keep_neg = actual_neg
                pos_frags = pos_cache[uid][:keep_pos] if keep_pos > 0 else []
                neg_frags = neg_cache[uid][:keep_neg] if keep_neg > 0 else []
                frags_to_merge = pos_frags + neg_frags

                if frags_to_merge:
                    merged = concat_dataproto_fragments(frags_to_merge)
                    selected_pool_batches.append(merged)
                    selected_count_by_uid[uid] = get_first_dim_size(merged)
                else:
                    print(f"[WARN] uid={uid} frags_to_merge is empty, cannot fallback")
            else:
                print(f"[WARN] uid={uid} not in pos_cache or neg_cache, cannot fallback")

        if not selected_pool_batches:
            raise RuntimeError(
                "No samples selected after early stopping. Check if threshold/rules are too strict or data is abnormal"
            )
        # gpt: 若最终没有任何样本被选出，则说明参数过严或奖励模型异常。

        # Concatenate all selected samples
        selected_batch = concat_dataproto_fragments(selected_pool_batches)
        # gpt: 合并所有被选中的样本 batch。

        # Align context fields to selected batch
        _context_src = context_batch if context_batch is not None else orig_prompt_batch
        ctx_rows = align_context_to_selected(selected_batch, _context_src)
        # gpt: 将上下文字段对齐（例如 prompt 文本、meta 信息等）。

        # Merge missing fields from context into selected batch
        merge_context_fields_into_batch(selected_batch, ctx_rows)
        # gpt: 将上下文中缺失的字段合并回选中 batch。

        final_batch = selected_batch

        # Ensure token_level_scores exists (fallback to token_level_rewards)
        if "token_level_scores" not in final_batch.batch and "token_level_rewards" in final_batch.batch:
            final_batch.batch["token_level_scores"] = final_batch.batch["token_level_rewards"]
        # gpt: 确保最终 batch 中存在 token 级别分数字段。

        # For GRPO with global stats, log pos/neg counts
        uid_to_pos_count = {uid: stats["total_pos"] for uid, stats in uid_full_stats.items()}
        uid_to_neg_count = {uid: stats["total_neg"] for uid, stats in uid_full_stats.items()}
        # gpt: 记录每个 prompt 的全局正负样本数量，用于 GRPO 统计。

        if not hasattr(final_batch, "meta_info") or final_batch.meta_info is None:
            final_batch.meta_info = {}
        final_batch.meta_info["grpo_uid_to_pos_count"] = uid_to_pos_count
        final_batch.meta_info["grpo_uid_to_neg_count"] = uid_to_neg_count

        # Validate that we maintained efficient TensorDict structure
        validate_tensordict_performance(final_batch, context="final_batch")
        # gpt: 验证 TensorDict 结构是否仍高效，防止拼接导致性能退化。

        return final_batch, rounds_info
        # gpt: 返回最终采样批次和每轮的统计信息。


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
                with marked_timer("step", timing_raw):
                    # generate a batch
                    # gpt: 这里是整个训练 step 的主计时上下文，用于记录整个步骤的耗时。
                    # gpt: 首先生成一个批次（batch）样本，后续根据算法配置分为两种情况：
                    # gpt: ① 开启 multiround_adaptive_downsampling（多轮自适应下采样）
                    # gpt: ② 未开启（标准单轮采样）

                    if self.config.algorithm.multiround_adaptive_downsampling:
                        # gpt: 开启多轮自适应下采样模式。
                        # gpt: 在这种模式下，生成过程会经历多个“回合（round）”，
                        # gpt: 每一轮根据 reward（奖励）信号筛选掉效果不佳的样本，仅保留潜力高的样本进入下一轮。
                        # gpt: 从而在生成阶段就提前过滤掉“低质量”样本，减少后续计算负担。
                        with marked_timer("gen_multi_round", timing_raw, color="red"):
                            final_batch, rounds_info = self._generate_multi_round_adaptive_downsampling(
                                orig_prompt_batch=gen_batch,
                                positive_threshold=self.config.algorithm.positive_threshold,
                                max_rounds=self.config.algorithm.max_rounds,
                                round_repeat=self.config.algorithm.round_repeat,
                                final_keep_per_prompt=self.config.actor_rollout_ref.rollout.n,
                                timing_raw=timing_raw,
                                context_batch=batch,
                            )

                        # gpt: 输出统计信息，每个 prompt 在多轮采样中经过了多少轮、被筛选多少次。
                        total_prompts = len(set(gen_batch.non_tensor_batch["uid"]))
                        print(
                            f"[Summary] prompts={total_prompts}, selected_rows={len(final_batch)}, "
                            f"max_rounds={self.config.algorithm.max_rounds}"
                        )
                        if rounds_info.get("per_round"):
                            for info in rounds_info["per_round"]:
                                print(
                                    f"  - round {info['round']}: active={info['active_prompts']}, "
                                    f"completed={info['completed']}, finished={info['finished_prompts']}, "
                                    f"time={info['sec']}s"
                                )

                        # gpt: 以下几项是对多轮采样过程的统计指标：
                        metrics["sampling/total_samples"] = np.sum(
                            [
                                (info["active_prompts"] * self.config.algorithm.round_repeat)
                                for info in rounds_info["per_round"]
                            ]
                        )
                        metrics["sampling/prompts_active_only_1st_round"] = rounds_info["per_round"][0][
                            "finished_prompts"
                        ]

                        if len(rounds_info["per_round"]) > 1:
                            metrics["sampling/prompts_active_after_1st_round"] = rounds_info["per_round"][1][
                                "active_prompts"
                            ] - (
                                rounds_info["per_round"][0]["active_prompts"]
                                - rounds_info["per_round"][-1]["finished_prompts"]
                            )
                        else:
                            metrics["sampling/prompts_active_after_1st_round"] = 0

                        metrics["sampling/prompts_no_positive_anywhere"] = (
                            rounds_info["per_round"][0]["active_prompts"]
                            - rounds_info["per_round"][-1]["finished_prompts"]
                        )
                        metrics["sampling/kept_samples"] = len(final_batch)
                        metrics["critic/real_reward"] = rounds_info["per_round"][0]["reward_mean"]
                        metrics["sampling/downsampled_samples"] = len(final_batch)
                        metrics["sampling/total_prompts"] = total_prompts

                        # gpt: 将多轮采样后的最终结果 batch 替换掉原始 batch。
                        batch = final_batch

                    else:
                        # gpt: 未开启 multiround_adaptive_downsampling。
                        # gpt: 这意味着采样过程只进行一次，不做多轮筛选。
                        # gpt: 所有 prompt 均生成 rollout 样本，无论质量高低，后续再通过 reward 与 advantage 估计修正。
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )

                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        # gpt: 如果使用 REMAX 优势估计器，则需要计算 baseline（基线 reward）
                        # gpt: 通过非采样模式（do_sample=False）生成参考输出作为基线比较。
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                        gen_baseline_batch
                                    )
                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output

                        # gpt: 对 batch 进行重复展开，使之与 rollout 的重复次数对齐。
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    # gpt: 如果没有 response_mask，就根据 response 的长度生成一个。
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # gpt: 为了在多卡（DP）训练中保持每个设备的 batch 大小一致，需要做平衡或 padding。
                    if self.config.trainer.balance_batch:
                        world_size = self.actor_rollout_wg.world_size
                        batch_size = len(batch)
                        if batch_size % world_size == 0:
                            self._balance_batch(batch, metrics=metrics)
                        else:
                            # Pad the batch to make it divisible by world_size
                            padding_needed = world_size - (batch_size % world_size)
                            print(f"Padding batch from {batch_size} to {batch_size + padding_needed} for balancing")

                            indices_to_repeat = random.choices(range(batch_size), k=padding_needed)
                            padding_batch = batch[indices_to_repeat]
                            batch = DataProto.concat([batch, padding_batch])

                            if hasattr(batch.batch, "__class__"):
                                batch_type = batch.batch.__class__.__name__
                                if "TensorDict" not in batch_type and "dict" in batch_type.lower():
                                    print(
                                        f"[perf_warn] After padding batch.batch is plain {batch_type}, may affect performance"
                                    )

                            self._balance_batch(batch, metrics=metrics)
                    batch.batch = batch.batch.contiguous()

                    # gpt: 计算全局 token 数量（例如总有效 token 数）
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        # gpt: 使用 reward model (RM) 或 reward_fn 来计算样本的奖励值。
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # gpt: 重新计算生成时的旧策略 log_prob，用于 PPO 等算法的 ratio 计算。
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            metrics.update(calculate_debug_metrics(batch))

                    # gpt: 如果开启参考策略（ref policy），则计算参考 log_prob，用于 KL 惩罚。
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # gpt: 计算 critic 的 value 估计值。
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # reward processing and downsampling already done in multi-round generation
                        # gpt: 注意：如果启用了 multiround_adaptive_downsampling，
                        # gpt: 奖励处理（reward processing）与样本下采样已在多轮阶段完成，
                        # gpt: 因此此处无需重复计算 reward tensor。
                        if not self.config.algorithm.multiround_adaptive_downsampling:
                            # gpt: 普通单轮模式下，在这里进行 reward 后处理。
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            # compute rewards. apply_kl_penalty if available
                            # gpt: 如果配置中启用了 KL 奖励惩罚（use_kl_in_reward），则此时叠加 KL。
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # gpt: 计算优势函数 advantage（即 reward - baseline）
                        # gpt: GRPO 中可选择是否按标准差归一化。
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # gpt: 更新 critic 网络
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # gpt: 如果 critic 已过热身阶段，则更新 actor。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # gpt: 如配置中开启 rollout 数据保存，则将生成样本、输出与得分写入文件。
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )
