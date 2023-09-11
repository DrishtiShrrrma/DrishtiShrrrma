# Portfolio

- [About Me](https://github.com/DrishtiShrrrma#about-me-%EF%B8%8F)
- [Achievements](https://github.com/DrishtiShrrrma#-achievements)
- [Certifications](https://github.com/DrishtiShrrrma#certifications)
- [Volunteer Experience](https://github.com/DrishtiShrrrma#-volunteer-experience)
- [NLP Projects](https://github.com/DrishtiShrrrma#-nlp-projects)
- [Audio Projects](https://github.com/DrishtiShrrrma#-audio-projects)
- [Reinforcement Learning Projects](https://github.com/DrishtiShrrrma#reinforcement-learning-projects)



## About Me 🙋‍♂️

Hello there! I'm **Drishti**. With a passion for technology and a knack for problem-solving, I've delved into various projects, from Natural Language Processing to Reinforcement Learning. (WIP)

## 🏆 Achievements

1. **Hugging Face Whisper Fine-tuning Event, Dec'22:**
   - Secured 1st position for [fine-tuned Whisper models](https://huggingface.co/collections/DrishtiSharma/whisper-fine-tuning-event-winning-models-64fe0b2e8710fc5ebddd27c0) for ASR task across 11 different low-resource languages, leveraging the Mozilla Common Voice 11 dataset. Languages included Azerbaijani, Breton, Hausa, Hindi, Kazakh, Lithuanian, Marathi, Nepali, Punjabi, Slovenian, and Serbian.
   - Models outperformed even the benchmarks set by OpenAI's Whisper research paper.
2. **Hugging Face Wav2Vec2 Fine-tuning Event, Feb'22:**
   - Attained 1st position with models fine-tuned for 7 distinct languages, namely: .
   - 
3. Analytics Vidhya Blogathon Winner - Best Guide
   
4. Analytics Vidhya Blogathon Winner - Best Article

5. Analytics Vidhya Blogathon Winner - Best Article -

## Certifications

1. [Hugging Face NLP Course Part-1 and Part-2](https://drive.google.com/file/d/1au4ozu8qrj0cV3391OwtyWknEXkxmA2K/view?usp=sharing).
2. [Hugging Face Deep Reinforcement Learning Course 2023](https://github.com/DrishtiShrrrma/huggingface-RL-course-2023#huggingface-rl-course-2023).
3. Hugging Face Audio Course

## 💪 Volunteer Experience

1. Reviewed an NLP research paper for [EMNLP - Aug'23](https://2023.emnlp.org/).
2. Trained, tested, and deployed TF-based models for [Keras at Hugging Face](https://huggingface.co/keras-io) - March'22.

# 🤖 NLP Projects

| Project Name       | Checkpoint & Code       | Key Highlights           | Performance Metrics                  | Blog                    | Demo   (WIP)               |
|--------------------|------------------|--------------------------|--------------------------|-------------------------|-----------------------|
| **Comparative Analysis of LoRA Parameters on Llama-2 with Flash Attention**        | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/enhancing-llama-2-a-study-of-flash-attention-and-lora-rank-64fca1772d20ced4e936f326); [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-dolly-15k-lora-parameter-analysis)     | While varying r and lora_dropout introduces subtle variations, lora_alpha has a more pronounced effect on training loss and inference efficiency.  | For lora_alpha=256: train_loss = 1.1358, training time=34min 52s, and inference time = 2.56s | [Blog](https://medium.com/@drishtisharma96505/comparative-analysis-of-lora-parameters-on-llama-2-with-flash-attention-574b913295d4)        |      |
| **Dissecting Llama-2-7b's Behavior with Varied Pretraining and Attention Mechanisms**          | [Hugging Face](); [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-alpaca-flash-atn-vs-atn-vs-tp)     | i) Flash Attention nearly halves the training time compared to Normal Attention. ii) Minimal difference in training loss across different pretraining_tp values. | train_loss=0.88    | [Blog](https://medium.com/@drishtisharma96505/dissecting-llama-2s-behavior-with-varied-pretraining-temperature-and-attention-mechanisms-18c47bd64dbf)        |      |
|        **Comparative Study: Training OPT-350M and GPT-2 Using Reward-Based Training**  | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/comparative-study-opt-350m-and-gpt-2-w-reward-based-training-64ff075830715b6d044fcb2d); [GitHub](https://github.com/DrishtiShrrrma/reward-modeling-rlhf-gpt2-vs-opt-350m)   | While opt-350m experienced a rapid initial decline in loss, GPT-2 showed a steadier descent but trained faster overall.  | Training Loss = 0.701 (GPT-2, OPT-350m) ;         Training Time: GPT-2 ---> 33:18, opt-350m ---> 1:23:54    | [Blog](https://medium.com/@drishtisharma96505/comparative-study-training-opt-350m-and-gpt-2-on-anthropics-hh-rlhf-dataset-using-reward-based-bd1050a5e6ac)        |      |
| Project 4          | Checkpoint 4     | Highlights of Project 4  | Metrics for Project 4    | [Blog 4 Link](#)        | [Demo 4 Link](#)      |
| Project 5          | Checkpoint 5     | Highlights of Project 5  | Metrics for Project 5    | [Blog 5 Link](#)        | [Demo 5 Link](#)      |
| Project 6          | Checkpoint 6     | Highlights of Project 6  | Metrics for Project 6    | [Blog 6 Link](#)        | [Demo 6 Link](#)      |
| Projec         | Checkpoint 7     | Highlights of Project 7  | Metrics for Project 7    | [Blog 7 Link](#)        | [Demo 7 Link](#)      |
| **Fine-Tuning Llama-2-7b on Databricks-Dolly-15k Dataset and Evaluating with BigBench-Hard**          | [Hugging Face](), [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-chat-hf-dolly-15k-w-bigbench-hard-evaluation/blob/main/llama_2_7b_chat_hf_databricks_dolly_15k.ipynb)     | While it demonstrated proficiency in handling general questions, there were instances where disparities emerged between the responses generated by the model and the anticipated answers, specifically during evaluations on BigBench-Hard questions. | train_loss = 2.343    | [Blog](https://github.com/DrishtiShrrrma/llama-2-7b-chat-hf-dolly-15k-w-bigbench-hard-evaluation#fine-tuning-llama-2-7b-on-databricks-dolly-15k-dataset-and-evaluating-with-bigbench-hard)        | [    |
| **Comparative Analysis of Adapter Vs Full Fine-Tuning- RoBERTa**          | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/adapter-64ff5f0f85a884a9649e9cf5), [GitHub]()     | Surprisingly and unexpectedly adapters performed better than the fully fine-tuned RoBERTa model, but, to have a concrete conclusion, more experiments must be conducted.  |    | [Blog](https://www.analyticsvidhya.com/blog/2023/04/training-an-adapter-for-roberta-model-for-sequence-classification-task)    |  |
| Project 10         | Checkpoint 10    | Highlights of Project 10 | Metrics for Project 10   | [Blog 10 Link](#)       |     |
| Project 11         | Checkpoint 11    | Highlights of Project 11 | Metrics for Project 11   | [Blog 11 Link](#)       |      |
| Project 12         | Checkpoint 12    | Highlights of Project 12 | Metrics for Project 12   | [Blog 12 Link](#)       |    |
| Project 13         | Checkpoint 13    | Highlights of Project 13 | Metrics for Project 13   | [Blog 13 Link](#)       |    |
| **codeBERT-based Password Strength Classifier**         | [Hugging Face](https://huggingface.co/DrishtiSharma/codebert-base-password-strength-classifier-normal-weight-balancing), [GitHub](https://github.com/DrishtiShrrrma/codebert-base-password-strength-classifier/)|Data Visualization Charts, Handled Imbalanced Data, Casing affected Password Strength  |  eval_Macro F1 = 0.9966, eval_Weighted Recall = 0.9977, eval_accuracy = 0.9977  | [Blog](https://github.com/DrishtiShrrrma/codebert-base-password-strength-classifier/blob/main/README.md)       |      |
| **BERT-base MCQA**       | [Hugging Face](https://huggingface.co/DrishtiSharma/bert-base-uncased-cosmos-mcqa), [GitHub](https://github.com/DrishtiShrrrma/bert-base-uncased-cosmos-mcqa)    | Overfitted Model, didn't perform really well | eval_loss= 2.03, eval_accuracy = 0.593   |        |      |
| **Fine-tuning 4-bit Llama-2-7b with Flash Attention using DPO**  | [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-dpo)   | **Training Halted Prematurely**  | **Training Halted Prematurely** | [Blog](https://medium.com/@drishtisharma96505/fine-tune-llama-2-7b-with-flash-attention-using-dpo-f989e7e6bfa4)       |     |


# 🎶 Audio Projects


| Project Name       | Checkpoint       | Key Highlights           | Metrics                  | Blog                    | Demo                  |
|--------------------|------------------|--------------------------|--------------------------|-------------------------|-----------------------|
| Project 1          | Checkpoint 1     | Highlights of Project 1  | Metrics for Project 1    | [Blog 1 Link](#)        | [Demo 1 Link](#)      |
| Project 2          | Checkpoint 2     | Highlights of Project 2  | Metrics for Project 2    | [Blog 2 Link](#)        | [Demo 2 Link](#)      |
| Project 3          | Checkpoint 3     | Highlights of Project 3  | Metrics for Project 3    | [Blog 3 Link](#)        | [Demo 3 Link](#)      |
| Project 4          | Checkpoint 4     | Highlights of Project 4  | Metrics for Project 4    | [Blog 4 Link](#)        | [Demo 4 Link](#)      |
| Project 5          | Checkpoint 5     | Highlights of Project 5  | Metrics for Project 5    | [Blog 5 Link](#)        | [Demo 5 Link](#)      |
| Project 6          | Checkpoint 6     | Highlights of Project 6  | Metrics for Project 6    | [Blog 6 Link](#)        | [Demo 6 Link](#)      |
| Project 7          | Checkpoint 7     | Highlights of Project 7  | Metrics for Project 7    | [Blog 7 Link](#)        | [Demo 7 Link](#)      |
| Project 8          | Checkpoint 8     | Highlights of Project 8  | Metrics for Project 8    | [Blog 8 Link](#)        | [Demo 8 Link](#)      |
| Project 9          | Checkpoint 9     | Highlights of Project 9  | Metrics for Project 9    | [Blog 9 Link](#)        | [Demo 9 Link](#)      |
| Project 10         | Checkpoint 10    | Highlights of Project 10 | Metrics for Project 10   | [Blog 10 Link](#)       | [Demo 10 Link](#)     |


# 🕹Reinforcement Learning Projects

| Index | Environment                      | Best Checkpoint                                                                                                 | mean_reward | Demo                                                                                                                             |
|-------|----------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------|
| 1     | LunarLander-v2                  | [Checkpoint](https://huggingface.co/DrishtiSharma/PPO-LunarLander-v2-12M-steps-successive-training)           | 280.89      | [Demo](https://huggingface.co/DrishtiSharma/PPO-LunarLander-v2-12M-steps-successive-training/resolve/main/replay.mp4)           |
| 2     | Taxi-v3                          | [Checkpoint](https://huggingface.co/DrishtiSharma/q-Taxi-v3-100000-episodes)                                 | 4.85        | [Demo](https://huggingface.co/DrishtiSharma/q-Taxi-v3-100000-episodes/resolve/main/replay.mp4)                                 |
| 3     | SpaceInvadersNoFrameskip-v4      | [Checkpoint](https://huggingface.co/DrishtiSharma/dqn-SpaceInvadersNoFrameskip-v4-2M-steps)                  | 502.78      | [Demo](https://huggingface.co/DrishtiSharma/dqn-SpaceInvadersNoFrameskip-v4-2M-steps/resolve/main/replay.mp4)                  |
| 4     | CartPole-v1                      | [Checkpoint](https://huggingface.co/DrishtiSharma/Reinforce-CartPole-v1-10k-steps)                           | 500         | [Demo](https://huggingface.co/DrishtiSharma/Reinforce-CartPole-v1-10k-steps/resolve/main/replay.mp4)                           |
| 5     | Pixelcopter-PLE-v0               | [Checkpoint](https://huggingface.co/DrishtiSharma/Reinforce-PixelCopter-1L)                                 | 25.01       | [Demo](https://huggingface.co/DrishtiSharma/Reinforce-PixelCopter-1L/resolve/main/replay.mp4)                                 |
| 6     | PandaReachDense                  | [Checkpoint](https://huggingface.co/DrishtiSharma/a2c-PandaReachDense-v2)                                    | -1.66       | [Demo](https://huggingface.co/DrishtiSharma/a2c-PandaReachDense-v2/resolve/main/replay.mp4)                                    |
| 7     | doom_health_gathering_supreme    | [Checkpoint](https://huggingface.co/DrishtiSharma/rl_course_vizdoom_health_gathering_supreme)                 | 7.12        | [Demo](https://huggingface.co/DrishtiSharma/rl_course_vizdoom_health_gathering_supreme/resolve/main/replay.mp4)                 |
| 8     | ML-Agents-SnowballTarget         | [Checkpoint](https://huggingface.co/DrishtiSharma/ppo-SnowballTarget)                                         | 0           | Demo                                                                                                                       |
| 9     | ML-Agents-Pyramids               | [Checkpoint](https://huggingface.co/DrishtiSharma/ppo-Pyramids)                                               | 0           | Demo                                                                                                                       |
| 10    | ML-Agents-SoccerTwos             | [Checkpoint](https://huggingface.co/DrishtiSharma/SoccerTwos-numlayers-16)                                   | 0           | Demo                                                                                                                       |



</details>

---

Thank you for taking the time to explore my journey! 👨‍💻









<!--
**DrishtiShrrrma/DrishtiShrrrma** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
