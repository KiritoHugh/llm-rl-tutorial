
                with marked_timer("step", timing_raw):
                    # generate a batch
                    if self.config.algorithm.multiround_adaptive_downsampling:
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

                        batch = final_batch

                    else:
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

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
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

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
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

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # reward processing and downsampling already done in multi-round generation
                        if not self.config.algorithm.multiround_adaptive_downsampling:
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
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
