# Installation
## Training environment
We use LLaMA-Factory to train our model, so you need to clone [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) first, then follow the instructions to install the training environment.

    conda create -n thinking_train python=3.10
    conda activate thinking_train
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation
    pip install deepspeed==0.16.9
    pip install tensorboard

## Inference environment
Reasoning on our model requires retrieval service and large language model service. We use [flashrag](https://github.com/RUC-NLPIR/FlashRAG.git) to build search service and use vllm to deploy KAG-Thinker-Model. You can install the environment as follows:

    conda create -n thinking_inference python=3.10
    conda activate thinking_inference
    git clone https://github.com/RUC-NLPIR/FlashRAG.git
    cd FlashRAG
    pip install -e .
    pip install vllm==0.6.6
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Training
Download [training dataset](https://huggingface.co/datasets/OpenSPG/KAG-Thinker-training-dataset) here, we name english training dataset "KAG_Thinker_en_train" , name chinese and engligh mixed dataset "KAG_Thinker_en_ch_train". The format of the training dataset, the dataset_info config and the hyperparameters for our model training are as follows:

<details>
<summary>Training Data Example</summary>

    {
        "source": "hotpotqa",
        "id": "train_68551",
        "messages": [
        {
            "role": "system",
            "content": "As you answer each question, you must provide a thought process and insert it between <think> and </think>."
        },
        {
            "role": "user",
            "content": "You are an expert in function calls, capable of accurately understanding function definitions and precisely decompose user queries to select appropriate functions to solve problems. The functions are as follows:\n\nFunction Name: Retrieval\nDescription: Search for SPO information. S stands for subject, O stands for object, represented as variable_name:entity_type[entity_name], where entity_name is an optional parameter required when there is a specific query entity; P represents predicate, i.e., relation or property, indicated as variable_name:edge_type or attribute_type. A unique variable name is assigned to each variable for subsequent reference. Note that S, P, O should not appear repeatedly within the same expression. When the variable refers to a previously defined variable, the variable name must match exactly, and only the variable name needs to be provided, with the entity type specified only upon first introduction.\nFunction Usage: Retrieval(s=s_alias:type['name'], p=p_alias:edge, o=o_alias:type['name'], p.prop='value', s.prop='value', o.prop='value')\n\nFunction Name: Math\nDescription: Perform calculations, which include set operations such as numerical calculations or sorting and counting. Content provides input information, which can be text or a referenced variable name. The target is the computational objective, usually the current subproblem. Math_alia is a variable name that represents its calculation result and can be referenced in subsequent actions.\nFunction Usage: Math(content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->math_alias\n\nFunction Name: Deduce\nDescription: Inference refers to the process of inferring search or calculation results to answer questions. op=judgement | entailment | rule | choice | multiChoice respectively represents true or false questions, implication reasoning (such as sentencing), fragment extraction, multiple choice questions, and multiple-choice questions. Content refers to questions, historical conversations, or search results, which can be text fragments or referred to by variable names. The target is the inference objective.\nFunction Usage: Deduce(op=judgement|entailment|extract|choice|multiChoice, content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->deduce_alias\n\nFunction Name: Output\nDescription: Directly output A, B, ... as the answer, where A and B are variable names referencing previous retrieval or calculation results.\nFunction Usage: Output(A,B,...)\n\nPlease, based on the definition of the above function, decompose the user question into one or multiple logical steps, outputting the execution plan for each step along with the corresponding action. Please note:\nStep: Accurately point out the logical thinking process of the question, and use #1 to refer to the solution result of Step1, #2 to refer to the solution result of Step2, and so on\nAction: Indicate exactly the function you selected and its parameters.\n\nQuestion:\n    Ernst John von Freyend served during World War II as the adjutant to a Field Marshal born in which year ?\nOutput:\n    "
        },
        {
            "role": "assistant",
            "content": "<think>This is a historical inquiry problem.Our investigation into the historical context of Ernst John von Freyend's service during World War II as an adjutant to a prominent Field Marshal unfolds through a series of interconnected steps. Initially, we identify the specific Field Marshal to whom von Freyend was assigned, a crucial piece of information that sets the stage for our inquiry. Subsequently, we delve into the biographical details of this military leader, pinpointing the year of his birth. This temporal data not only enriches our understanding of the historical period in question but also serves as the final piece of our narrative, allowing us to connect the dots between Ernst John von Freyend's military career and the broader historical events of World War II. By meticulously following this plan, we ensure that our answer is both accurate and informative, providing a comprehensive response to the original question.</think>\n\n<answer>Step1: The Field Marshal Ernst John von Freyend served as an adjutant to?\nAction1: Retrieval(s=s1:militaryPersonnel[`Ernst John von Freyend`], p=p1:servedAsAdjutantTo, o=o1:fieldMarshal)\nStep2: The time of birth of #1?\nAction2: Retrieval(s=o1, p=p2:birthTime, o=o2:birthTime)\nStep3: Output #2\nAction3: Output(o2)</answer>"
        },
        {
            "role": "user",
            "content": "Can you answer the following questions step by step? If you can, wrap your answer with <answer>\\boxed{your answer}</answer>. If you can't, just reply that based on my internal knowledge, I can't answer this question, I need to retrieve external knowledge. \nQuestion: \nStep1: The Field Marshal Ernst John von Freyend served as an adjutant to?\nAction1: Retrieval(s=s1:militaryPersonnel[`Ernst John von Freyend`], p=p1:servedAsAdjutantTo, o=o1:fieldMarshal)"
        },
        {
            "role": "assistant",
            "content": "<think>based on my internal knowledge, I can't answer this question, I need to retrieve external knowledge.</think>\n<search>Step1: The Field Marshal Ernst John von Freyend served as an adjutant to?\nAction1: Retrieval(s=s1:militaryPersonnel[`Ernst John von Freyend`], p=p1:servedAsAdjutantTo, o=o1:fieldMarshal)</search>"
        },
        {
            "role": "user",
            "content": "<references>\"Ernst John von Freyend\"\nof the war. After surrendering with Keitel, he was briefly interrogated by the Americans who decided that he was not involved in military decision making and acted mainly as Keitel's valet. In the early 1950s, he became an employee of Gehlen Organization. Ernst John von Freyend has been portrayed by the following actors in film productions. Ernst John von Freyend Ernst John von Freyend (25 March 1909– 24 March 1980) was a German Oberkommando der Wehrmacht officer who served during World War II as the adjutant to Field Marshal Wilhelm Keitel. He is notable for unwittingly helping to place the\n\n\"Ernst John von Freyend\"\nErnst John von Freyend Ernst John von Freyend (25 March 1909– 24 March 1980) was a German Oberkommando der Wehrmacht officer who served during World War II as the adjutant to Field Marshal Wilhelm Keitel. He is notable for unwittingly helping to place the 20 July plot bomb that was intended to kill Adolf Hitler. On 20 July 1944, he held the rank of Major and arrived at the Wolf's Lair headquarters in Rastenburg, East Prussia for a situation conference attended by Hitler. Also there as a subordinate to Field Marshal Keitel was Colonel Claus von Stauffenberg who arrived clutching\n\n\"Franz Conrad von Hötzendorf\"\nFranz Conrad von Hötzendorf \"\"K.u.k. Feldmarschall\"\" Franz Xaver Joseph Graf Conrad von Hötzendorf (; 11 November 1852 – 25 August 1925), sometimes anglicised as Hoetzendorf, was an Austrian general who played a central role in World War I. He served as Field Marshal and Chief of the General Staff of the military of the Austro-Hungarian Army and Navy from 1906 to 1917. He was in charge during the July Crisis of 1914 that caused World War I. For years he had repeatedly called for preemptive war against Serbia to rescue the multiethnic Habsburg Empire, which was, he believed, nearing disintegration.\n\n</references>"
        },
        {
            "role": "assistant",
            "content": "<think> Having found the information, I can now ascertain that Ernst John von Freyend served as an adjutant to Field Marshal Wilhelm Keitel. This is based on the information provided in the search results.</think>\n<search>Step1:  Who was Field Marshal Wilhelm Keitel? \nAction1: Retrieval(s=s1:militaryPersonnel[`Field Marshal Wilhelm Keitel`], p=p1:title, o=o1:title)</search>"
        },
        {
            "role": "user",
            "content": "<references>\"Wilhelm Keitel\"\nWilhelm Keitel Wilhelm Bodewin Johann Gustav Keitel (22 September 1882 – 16 October 1946) was a German field marshal who served as Chief of the Armed Forces High Command ( or OKW) in Nazi Germany during World War II. According to David Stahel, Keitel was \"\"well known and [...] reviled as Hitler’s dependable mouthpiece and habitual yes-man\"\" among his military colleagues. Following the war, Keitel was charged with war crimes and crimes against humanity by the International Military Tribunal at Nuremberg. He was found guilty, sentenced to death and executed by hanging in 1946. He was the third highest-ranking German\n\n\"Wilhelm Keitel\"\nCommand, 1938–1945\"\" edited by Walter Görlitz, . Notes \"\"Bibliography\"\"' Wilhelm Keitel Wilhelm Bodewin Johann Gustav Keitel (22 September 1882 – 16 October 1946) was a German field marshal who served as Chief of the Armed Forces High Command ( or OKW) in Nazi Germany during World War II. According to David Stahel, Keitel was \"\"well known and [...] reviled as Hitler’s dependable mouthpiece and habitual yes-man\"\" among his military colleagues. Following the war, Keitel was charged with war crimes and crimes against humanity by the International Military Tribunal at Nuremberg. He was found guilty, sentenced to death and executed by\n\n\"Bodewin Keitel\"\nBodewin Keitel Bodewin Keitel (1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office. Bodewin Keitel was born on 25 December 1888 in Helmscherode, the son of a landowner. He was the brother of Field Marshal Wilhelm Keitel, later head of the High Command of the Wehrmacht. Bodewin joined the army in 1909 and served during World War I. After the end of World War I, he was retained in the Reichswehr, eventually reaching the rank of colonel. In 1937, he was appointed as the Chief of Training Department (\"\"T4\"\")\n\n</references>"
        },
        {
            "role": "assistant",
            "content": "<think> Now that I have identified the Field Marshal Ernst John von Freyend's adjutant, I can provide the final answer.</think>\n<answer> \\boxed{Field Marshal Wilhelm Keitel} </answer>"
        },
        {
            "role": "user",
            "content": "Can you answer the following questions step by step? If you can, wrap your answer with <answer>\\boxed{your answer}</answer>. If you can't, just reply that based on my internal knowledge, I can't answer this question, I need to retrieve external knowledge. \nQuestion: \nStep2: The time of birth of Field Marshal Wilhelm Keitel?\nAction2: Retrieval(s=o1, p=p2:birthTime, o=o2:birthTime)"
        },
        {
            "role": "assistant",
            "content": "<think>The answer to the question is \\boxed{1882-01-10}\nAlright, let's analyze step-by-step whether the answers above are correct?Let's approach this step-by-step:\n\n1. The question asks for the time of birth of Field Marshal Wilhelm Keitel.\n2. The provided answer is a date: 1882-01-10.\n3. The date format (YYYY-MM-DD) is correct.\n4. However, the question specifically asks for the \"time\" of birth, not just the date.\n5. The provided answer only gives the date, not the time.\n6. While the date is correct, the answer does not fully address the question as asked.\n\nTherefore, the answer is:</think>\n\n<search>Step2: The time of birth of Field Marshal Wilhelm Keitel?\nAction2: Retrieval(s=o1, p=p2:birthTime, o=o2:birthTime)</search>"
        },
        {
            "role": "user",
            "content": "<references>\"Bodewin Keitel\"\nBodewin Keitel Bodewin Keitel (1888 – 1953) was a German general during World War II who served as head of the Army Personnel Office. Bodewin Keitel was born on 25 December 1888 in Helmscherode, the son of a landowner. He was the brother of Field Marshal Wilhelm Keitel, later head of the High Command of the Wehrmacht. Bodewin joined the army in 1909 and served during World War I. After the end of World War I, he was retained in the Reichswehr, eventually reaching the rank of colonel. In 1937, he was appointed as the Chief of Training Department (\"\"T4\"\")\n\n\"Wilhelm Keitel\"\nCommand, 1938–1945\"\" edited by Walter Görlitz, . Notes \"\"Bibliography\"\"' Wilhelm Keitel Wilhelm Bodewin Johann Gustav Keitel (22 September 1882 – 16 October 1946) was a German field marshal who served as Chief of the Armed Forces High Command ( or OKW) in Nazi Germany during World War II. According to David Stahel, Keitel was \"\"well known and [...] reviled as Hitler’s dependable mouthpiece and habitual yes-man\"\" among his military colleagues. Following the war, Keitel was charged with war crimes and crimes against humanity by the International Military Tribunal at Nuremberg. He was found guilty, sentenced to death and executed by\n\n\"Wilhelm Keitel\"\nofficer to be tried at Nuremberg. Keitel was born in the village of Helmscherode near Gandersheim in the Duchy of Brunswick, the eldest son of Carl Keitel (1854–1934), a middle class landowner, and his wife Apollonia Vissering (1855–1888). After he completed his education at gymnasium in Göttingen, his plan to take over his family's estates foundered on his father's resistance. Instead, he embarked on a military career in 1901, becoming an officer cadet of the Prussian Army. As a commoner he did not join the cavalry, but the mounted 46th Lower-Saxon Field Artillery Regiment in Wolfenbüttel, serving as adjutant from\n\n</references>"
        },
        {
            "role": "assistant",
            "content": "<think> The search results indicate that Wilhelm Keitel was born on 22 September 1882. </think>\n<search>Step1:  Wilhelm Keitel birth date \nAction1: Retrieval(s=s1:person[`Wilhelm Keitel`], p=p1:birthDate, o=o1:date)</search>"
        },
        {
            "role": "user",
            "content": "<references>\"Wilhelm Keitel\"\nWilhelm Keitel Wilhelm Bodewin Johann Gustav Keitel (22 September 1882 – 16 October 1946) was a German field marshal who served as Chief of the Armed Forces High Command ( or OKW) in Nazi Germany during World War II. According to David Stahel, Keitel was \"\"well known and [...] reviled as Hitler’s dependable mouthpiece and habitual yes-man\"\" among his military colleagues. Following the war, Keitel was charged with war crimes and crimes against humanity by the International Military Tribunal at Nuremberg. He was found guilty, sentenced to death and executed by hanging in 1946. He was the third highest-ranking German\n\n\"Wilhelm Keitel\"\n1908. On 18 April 1909, Keitel married Lisa Fontaine, a wealthy landowner's daughter at Wülfel near Hanover. Together they had six children, one of whom died in infancy. His eldest son, Karl-Heinz Keitel (1914–1968), went on to serve as a divisional commander (Obersturmbannführer (Lieutenant Colonel)) in the Waffen-SS. During World War I, Keitel served on the Western Front with his artillery regiment and took part in the fighting in Flanders, where he was severely wounded in his right forearm by a shell fragment. Promoted to captain, Keitel quickly recovered, and in 1915 posted to the General Staff of the 19th\n\n\"Wilhelm Keitel\"\nCommand, 1938–1945\"\" edited by Walter Görlitz, . Notes \"\"Bibliography\"\"' Wilhelm Keitel Wilhelm Bodewin Johann Gustav Keitel (22 September 1882 – 16 October 1946) was a German field marshal who served as Chief of the Armed Forces High Command ( or OKW) in Nazi Germany during World War II. According to David Stahel, Keitel was \"\"well known and [...] reviled as Hitler’s dependable mouthpiece and habitual yes-man\"\" among his military colleagues. Following the war, Keitel was charged with war crimes and crimes against humanity by the International Military Tribunal at Nuremberg. He was found guilty, sentenced to death and executed by\n\n</references>"
        },
        {
            "role": "assistant",
            "content": "<think> The search results confirm that Wilhelm Keitel was born on 22 September 1882. </think>\n<answer> The time of birth of Field Marshal Wilhelm Keitel is \\[ \\boxed{22 September 1882} \\] </answer>."
        },
        {
            "role": "user",
            "content": "Answer the last question based on question, the first n sub-questions and their answers (indicated by #n), and wrap the result with <answer>\\boxed{your answer}</answer>. \nquestion:\nErnst John von Freyend served during World War II as the adjutant to a Field Marshal born in which year ?\nsub-questions:\nThe Field Marshal Ernst John von Freyend served as an adjutant to?\n#1: Field Marshal Wilhelm Keitel\nThe time of birth of Field Marshal Wilhelm Keitel?\n#2: 22 September 1882\n\nlast-question:\nOutput #2\nanswer:\n"
        },
        {
            "role": "assistant",
            "content": "<think>I think I can answer this question now</think>\n\n<answer>\\boxed{1882}</answer>."
        }
        ]
    }

</details>

# Dataset Convert
To train with LLaMA-Factory, the JSONL data format needs to be converted into a single large JSON file. The converted format should be:
```json
[
  {	
    "source": "nq",
	"id": "xxx",
	"messages": [...]
  },
  {	
    "source": "nq",
	"id": "xxx",
	"messages": [...]
  },...
]
```

<details>
<summary>Dataset_info</summary>

    "KAG_Thinker_en_train_convert": {
        "file_name": "KAG_Thinker_en_train_convert.json",
        "formatting": "sharegpt",
        "columns": {
                "messages": "messages"
        },
        "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
        }        
    }
</details>

<details>
<summary>Hyperparameters</summary>

    model_name_or_path: Qwen/Qwen2.5-7B-Instruct
    trust_remote_code: true

    stage: sft
    do_train: true
    finetuning_type: full
    deepspeed: examples/deepspeed/ds_z3_config.json

    dataset: musique_multi_search_logic_form_train_top3_en
    template: qwen
    cutoff_len: 16384
    max_samples: 50000000
    overwrite_cache: true
    preprocessing_num_workers: 64
    dataloader_num_workers: 1

    output_dir: saves/qwen2.5/full/sft
    logging_steps: 10
    save_strategy: epoch
    save_steps: 300
    plot_loss: true
    overwrite_output_dir: true
    save_only_model: true
    report_to: tensorboard

    per_device_train_batch_size: 2
    gradient_accumulation_steps: 2
    learning_rate: 5.0e-6
    num_train_epochs: 5.0
    lr_scheduler_type: linear
    warmup_ratio: 0.06
    bf16: true
    ddp_timeout: 10800
    resume_from_checkpoint: null

    val_size: 0.02
    per_device_eval_batch_size: 2
    eval_strategy: epoch
    eval_steps: 300
</details>

Our model was trained using the above training dataset, training dataset_info and training hyperparameters, we name it KAG-Thinker-Model. You can download [KAG-Thinker-en](https://huggingface.co/OpenSPG/KAG-Thinker-en-7b-instruct) here.

# Inference
The thinker model will decide whether to use external search tools based on the degree of knowledge it has mastered. Therefore, the complete reasoning process requires a search service and a thinker LLM. Follow the steps below to start the search service and LLM service before reasoning. The format of the test dataset is as follows:

    {"id": "test_3609", "question": "when did computer become widespread in homes and schools", "golden_answers": ["1980s"]}


## Start retrieval service
Before starting the retrieval service, you need to configure the parameters in retriever_config.yaml. "gpu_id" is the gpu required for the retrieval service; [retrieval_method](https://huggingface.co/intfloat/e5-base-v2) is the vector model of the retrieval service; [index_path](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip) is the constructed index library file; [corpus_path](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus) is the original file of the index library. Now start retrieval service as follow:

    cd retrieval
    python retriever_serving.py
    
## Start LLM service

    model_path=your_model_path
    python -m vllm.entrypoints.openai.api_server --served-model-name kag-thinker-multiturn --model ${model_path} --trust-remote-code --host 0.0.0.0 --port 12389 --max-model-len 32768 --tensor-parallel-size 4 --gpu_memory_utilization 0.9 --enforce-eager --dtype auto --enable-prefix-caching --swap-space 4

## Start Inference 

    cd inference
    dataset=your_inference_dataset_path
    # english version inference
    python inference_en.py --filename ${dataset} 
    # chinese version inference
    python inference_ch.py --filename ${dataset}

# Multi-language version
We not only provide english version [KAG-Thinker-en](https://huggingface.co/OpenSPG/KAG-Thinker-en-7b-instruct), but also provide a multi-language version [KAG-Thinker-en-ch](https://huggingface.co/OpenSPG/KAG-Thinker-en-ch-7b-instruct). The evaluation EM results of the two versions are as follows:

|                   | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | AVG |
| ----------------- | ---------- | -----------| ---------- | ---------- | -----------| ---------- | ---------- | -----------|
| KAG-Thinker-en    | 0.455   | 0.622  | 0.480  | 0.424   | 0.473   | 0.214   | 0.496   |  0.452  |
| KAG-Thinker-en-ch | 0.458   | 0.635   | 0.475   | 0.419  | 0.477  | 0.224   | 0.456  | 0.449   |

# Citiation
If you want to know more details, see our technical report.

    @misc{zhang2025kagthinkerteachinglargelanguage,
      title={KAG-Thinker: Teaching Large Language Models to Think with Human-like Reasoning Process}, 
      author={Dalong Zhang and Jun Xu and Jun Zhou and Lei Liang and Lin Yuan and Ling Zhong and Mengshu Sun and Peilong Zhao and QiWei Wang and Xiaorui Wang and Xinkai Du and YangYang Hou and Yu Ao and ZhaoYang Wang and Zhengke Gui and ZhiYing Yi and Zhongpu Bo},
      year={2025},
      eprint={2506.17728},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.17728}, 
    }