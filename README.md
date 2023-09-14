# Portfolio 

- [About Me](https://github.com/DrishtiShrrrma#about-me-%EF%B8%8F)
- [Achievements](https://github.com/DrishtiShrrrma#-achievements)
- [Certifications](https://github.com/DrishtiShrrrma#certifications)
- [Volunteer Experience](https://github.com/DrishtiShrrrma#-volunteer-experience)
- [NLP Projects](https://github.com/DrishtiShrrrma#-nlp-projects)
- [Audio Projects](https://github.com/DrishtiShrrrma#-audio-projects)
- [Reinforcement Learning Projects](https://github.com/DrishtiShrrrma#reinforcement-learning-projects)



## 🙋‍♂️About Me 

Hello there! I'm **Drishti**. With a passion for technology and a knack for problem-solving, I've delved into various projects, from Natural Language Processing to Audio DL and Reinforcement Learning.

I started my career with CPA Global (now Clarivate), Noida, where I worked for four years as an IP Researcher and IP Consultant respectively. This deep dive into patents not only honed my analytical skills but also introduced me to the vast expanse of AI. However, as time passed, a restless quest for greater purpose and multiple unexpected twists in life pushed me to re-invent and rebuild my life.

After segueing into the Data Science domain I've actively engaged with the [Hugging Face](https://huggingface.co/DrishtiSharma/) Hub's open-source initiatives and disseminated my research insights through [Medium](https://medium.com/@drishtisharma96505) and [Analytics Vidhya](https://www.analyticsvidhya.com/blog/author/drishti_sharma/).

## 🏆 Achievements

1. **Hugging Face Whisper Fine-tuning Event, Dec'22:**
   - Secured 1st position for [fine-tuned Whisper models](https://huggingface.co/collections/DrishtiSharma/whisper-fine-tuning-event-winning-models-64fe0b2e8710fc5ebddd27c0) for ASR task across 11 different low-resource languages, leveraging the Mozilla Common Voice 11 dataset. Languages included Azerbaijani, Breton, Hausa, Hindi, Kazakh, Lithuanian, Marathi, Nepali, Punjabi, Slovenian, and Serbian.
   - Models outperformed even the benchmarks set by OpenAI's Whisper research paper.
2. **Hugging Face Wav2Vec2 Fine-tuning Event, Feb'22:**
   - Attained 1st position with models fine-tuned for 7 distinct languages.
   - 
3. **Secured 3rd place in Analytics Vidhya Blogathon'26 for 'Best Guide'.**
   
4. **Ranked 3rd in Analytics Vidhya Blogathon'26 for 'Best Article'.**

5. **Analytics Vidhya Blogathon Winner - Best Article**

## 🥇 Certifications

1. [Hugging Face NLP Course Part-1 and Part-2](https://drive.google.com/file/d/1au4ozu8qrj0cV3391OwtyWknEXkxmA2K/view?usp=sharing).
2. [Hugging Face Deep Reinforcement Learning Course 2023](https://github.com/DrishtiShrrrma/huggingface-RL-course-2023#huggingface-rl-course-2023).
3. Hugging Face Audio Course

## 💪 Volunteer Experience

1. Reviewed an NLP research paper for [EMNLP - Aug'23](https://2023.emnlp.org/).
2. Trained, tested, and deployed TF-based models for [Keras at Hugging Face](https://huggingface.co/keras-io) - March'22.

# 🤖 NLP Projects

| Project Name       | Checkpoint & Code       | Key Highlights           | Performance Metrics                  | Blog                    | Demo   (WIP)               |
|--------------------|------------------|--------------------------|--------------------------|-------------------------|-----------------------|
| **Comparative Analysis of LoRA Parameters on Llama-2 with Flash Attention**        | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/enhancing-llama-2-a-study-of-flash-attention-and-lora-rank-64fca1772d20ced4e936f326); [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-dolly-15k-lora-parameter-analysis)     | Optimal performance is achieved at r=32; varying lora_dropout yields stable training loss but inconsistent inference times, while increasing lora_alpha improves training without sacrificing efficiency.  | For lora_alpha=256: train_loss = 1.1358, training time=34min 52s, and inference time = 2.56s | [Blog](https://medium.com/@drishtisharma96505/comparative-analysis-of-lora-parameters-on-llama-2-with-flash-attention-574b913295d4)        |      |
| **Dissecting Llama-2-7b's Behavior with Varied Pretraining Temperature and Attention Mechanisms**          | [Hugging Face](); [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-alpaca-flash-atn-vs-atn-vs-tp)     | i) Flash Attention nearly halves the training time compared to Normal Attention. ii) Minimal difference in training loss across different pretraining_tp values. | train_loss=0.88    | [Blog](https://medium.com/@drishtisharma96505/dissecting-llama-2s-behavior-with-varied-pretraining-temperature-and-attention-mechanisms-18c47bd64dbf)        |      |
|        **Comparative Study: Training OPT-350M and GPT-2 Using Reward-Based Training**  | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/comparative-study-opt-350m-and-gpt-2-w-reward-based-training-64ff075830715b6d044fcb2d); [GitHub](https://github.com/DrishtiShrrrma/reward-modeling-rlhf-gpt2-vs-opt-350m)   | While opt-350m experienced a rapid initial decline in loss, GPT-2 showed a steadier descent but trained faster overall.  | Training Loss = 0.701 (GPT-2, OPT-350m) ;         Training Time: GPT-2 ---> 33:18, opt-350m ---> 1:23:54    | [Blog](https://medium.com/@drishtisharma96505/comparative-study-training-opt-350m-and-gpt-2-on-anthropics-hh-rlhf-dataset-using-reward-based-bd1050a5e6ac)        |      |
| **Unraveling the Dual Impact: Batch Size and Mixed Precision on DistilBERT’s Performance in Language Detection**         | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/studying-impact-of-batch-size-and-mixed-precision-64fff55f70b6b05c5ac53e50), [GitHub](https://github.com/DrishtiShrrrma/distilbert-base-language-detection-analysis)    | Performance metrics, such as training and validation losses, exhibit a subtle deterioration with very large batch sizes, suggesting possible coarse gradient approximations. Furthermore, utilizing fp16 enhances computational speed across different batch sizes while maintaining comparable accuracy metrics  |  | [Blog](https://medium.com/@drishtisharma96505/unraveling-the-dual-impact-batch-size-and-mixed-precision-on-distilberts-performance-in-language-56ab2d2b20b8)        |      |
| **Analyzing the Impact of lora_alpha on Llama-2 Quantized with GPTQ**| [Hugging Face](https://huggingface.co/collections/DrishtiSharma/studying-impact-of-lora-alpha-on-llama-2-quantized-with-gptq-65033862e88eb2d0d5019e09), [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-chat-gptq-english-quotes-lora-alpha-analysis)|At lora_alpha 32, optimal training (3.8675) and validation losses (4.2374) were achieved, but values beyond this showed decreased performance and potential overfitting, while runtimes remained consistent. | At lora_alpha=32: train_loss=3.8675, val_loss = 4.2374|[Blog](https://medium.com/@drishtisharma96505/analyzing-the-impact-of-lora-alpha-on-llama-2-quantized-with-gptq-f01e8e8ed8fd) |
| **Comprehensive Evaluation of Various Transformer Models in Detecting Normal, Hate, and Offensive Texts**          |[GitHub](https://github.com/DrishtiShrrrma/multiclass-classification-transformer-comparison)    | bert-base-uncased stands out as a top performer that balances efficiency and precision. It did even better than roberta-large!  | Please refer to the blog for the same, tough to encapsulate here    | [Blog](https://medium.com/@drishtisharma96505/comprehensive-evaluation-of-various-transformer-models-in-detecting-normal-hate-and-offensive-a63c5c5226f)        |       |
| **Unveiling the Impact of Weight Decay on MBart-large-50 for English-Spanish Translation**         | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/impact-of-weight-decay-on-mbart-large-50-for-en-es-64ff84133587d3ebfd77fcae); [GitHub](https://github.com/DrishtiShrrrma/mbart-50-en-to-es-translation-weight-decay-analysis)     | Weight-decay shows only a muted influence on the MBart-50 model’s English-Spanish translation performance.  | BLEU = 45.08, RougeLsum=0.684    | [Blog](https://medium.com/@drishtisharma96505/unveiling-the-impact-of-weight-decay-on-mbart-large-50-for-english-spanish-translation-1c2fe3fb854d)        |      |
| **Fine-Tuning Llama-2-7b on Databricks-Dolly-15k Dataset and Evaluating with BigBench-Hard**          | [Hugging Face](), [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-chat-hf-dolly-15k-w-bigbench-hard-evaluation/blob/main/llama_2_7b_chat_hf_databricks_dolly_15k.ipynb)     | While it demonstrated proficiency in handling general questions, there were instances where disparities emerged between the responses generated by the model and the anticipated answers, specifically during evaluations on BigBench-Hard questions. | train_loss = 2.343    | [Blog](https://github.com/DrishtiShrrrma/llama-2-7b-chat-hf-dolly-15k-w-bigbench-hard-evaluation#fine-tuning-llama-2-7b-on-databricks-dolly-15k-dataset-and-evaluating-with-bigbench-hard)        |    |
| **Comparative Analysis of Adapter Vs Full Fine-Tuning- RoBERTa**          | [Hugging Face](https://huggingface.co/collections/DrishtiSharma/adapter-64ff5f0f85a884a9649e9cf5), [GitHub]()     | Surprisingly and unexpectedly adapters performed better than the fully fine-tuned RoBERTa model, but, to have a concrete conclusion, more experiments must be conducted.  |    | [Blog](https://www.analyticsvidhya.com/blog/2023/04/training-an-adapter-for-roberta-model-for-sequence-classification-task)    |  |
| **codeBERT-based Password Strength Classifier**         | [Hugging Face](https://huggingface.co/DrishtiSharma/codebert-base-password-strength-classifier-normal-weight-balancing), [GitHub](https://github.com/DrishtiShrrrma/codebert-base-password-strength-classifier/)|Data Visualization Charts, Handled Imbalanced Data, Casing affected Password Strength  |  eval_Macro F1 = 0.9966, eval_Weighted Recall = 0.9977, eval_accuracy = 0.9977  | [Blog](https://github.com/DrishtiShrrrma/codebert-base-password-strength-classifier/blob/main/README.md)       |      |
| **BERT-base MCQA**       | [Hugging Face](https://huggingface.co/DrishtiSharma/bert-base-uncased-cosmos-mcqa), [GitHub](https://github.com/DrishtiShrrrma/bert-base-uncased-cosmos-mcqa)    | Overfitted Model, didn't perform really well | eval_loss= 2.03, eval_accuracy = 0.593   |        |      |
| **Fine-tuning 4-bit Llama-2-7b with Flash Attention using DPO**  | [GitHub](https://github.com/DrishtiShrrrma/llama-2-7b-dpo)   | **Training Halted Prematurely**  | **Training Halted Prematurely** | [Blog](https://medium.com/@drishtisharma96505/fine-tune-llama-2-7b-with-flash-attention-using-dpo-f989e7e6bfa4)       |     |
|**Sentence-t5-large Quora Text Similarity Checker**|[Hugging Face](), [GitHub](https://github.com/DrishtiShrrrma/sentence-t5-large-quora-text-similarity)||||
|**Stable Diffusion Prompt Generator**|[Hugging Face](), [GitHub](https://github.com/DrishtiShrrrma/stablediffusion-prompt-generator)||||


# 🎶 Audio Projects


| Project Name       | Checkpoint       | Metrics                  | Blog                    | Demo                  |
|--------------------|------------------|--------------------------|-------------------------|-----------------------|
| ASR using Whisper        | [Hugging Face](), [GitHub]()    |   |     |         |       
| Text-to-Speech Using SpeechT5         | [Hugging Face](https://huggingface.co/DrishtiSharma/speecht5_finetuned_voxpopuli_es_20k_steps_bs_8), [GitHub](https://github.com/DrishtiShrrrma/speechT5-spanish-tts)  |||||
|DistilHuBERT fine-tuned on GTZAN for Audio Classification Task   | [Hugging Face](https://huggingface.co/DrishtiSharma/distilhubert-finetuned-gtzan-bs-4-fp16-false), [GitHub](https://github.com/DrishtiShrrrma/huggingface-audio-course/blob/main/unit4/distilhubert_finetuned_gtzan_bs_16.ipynb)     |         |     
|**Wav2Vec2 fine-tuned on MESD Dataset for Emotion Classification**|[Hugging Face](https://huggingface.co/DrishtiSharma/wav2vec2-base-finetuned-sentiment-mesd-v9)||Accuracy = 91.54%||
|**Time-stamp Prediction using Whisper**|[GitHub](https://github.com/DrishtiShrrrma/huggingface-audio-course/blob/main/unit5/Timestamp_Prediction_Using_Whisper_and_HF_Pipeline.ipynb)||||



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
