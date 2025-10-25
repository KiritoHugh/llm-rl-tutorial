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
