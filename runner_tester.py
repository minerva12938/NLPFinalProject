import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import random

NUM_PREPROCESSING_WORKERS = 2

# creates contrast sets for the given premises by perterbing
def prepare_contrast_sets(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    colors = []
    people = []
    animals = []
    nums = []
    time = []
    is_not = []
    are_not = []
    vehicles = []
    temp = []
    prepositions = []

    colors = ['black', 'blue', 'red', 'orange', 'yellow', 'green', 'purple', 'white', 'pink', 'brown']
    people = ['child', 'man', 'woman', 'male', 'female', 'men', 'women', 'males', 'women', 'kid', 'adult', 'kids', 'elders', 'boy', 'girl', 'boys', 'girls', 'guys', 'gals', 'gentleman', 'lady', 'gentlemen', 'ladies', 'person', 'people']
    animals = ['dog', 'cat', 'bird', 'fish', 'squirrel', 'bat', 'horse', 'pony', 'dogs', 'cats', 'birds', 'squirrels', 'bats', 'horses', 'ponies']
    nums = ['2', '3', '4', '5', '6', '7', '8', '9', '10' 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'some', 'many']
    time = ['early', 'late', 'night', 'morning', 'afternoon', 'evening']
    is_not = ['is', 'is not']
    are_not = ['are', 'are not', 'have', 'have not', 'was', 'was not']
    vehicles = ['car', 'bus', 'truck', 'scooter', 'bike', 'bicycle', 'tricycle', 'motorbike', 'unicycle', 'cars', 'buses', 'trucks', 'scooters', 'bikes', 'bicycles', 'tricycles', 'motorbikes', 'unicycles']
    temp = ['hot', 'cold', 'cool', 'humid', 'chilly', 'freezing', 'blistering', 'sweltering']
    prepositions = ['around', 'in', 'on', 'above', 'under', 'outside', 'inside', 'with', 'without']

    contrast_set = []
    hypotheses = []

    # num_exs = 10
    num_exs = len(examples['premise']) - 1

    i = 0
    while i < num_exs:
      curr_ex = examples['premise'][i]
      curr_hyp = examples['hypothesis'][i]
      all_exs = []
      hyps = []
      all_exs.append(curr_ex)
      hyps.append(curr_hyp)
      if i < num_exs - 1:
        next_ex = examples['premise'][i + 1]
        next_hyp = examples['hypothesis'][i + 1]
        i += 1
      while next_ex == curr_ex and i < num_exs:
        all_exs.append(next_ex)
        hyps.append(next_hyp)
        i += 1
        next_ex = examples['premise'][i]
        next_hyp = examples['hypothesis'][i]
      contrast_set.append(all_exs)
      hypotheses.append(hyps)

    # print(contrast_set)

    perturbations = []
    perturbed_hyps = []
    to_break = False
    group_num = 0
    for group in contrast_set:
        # print("group: " + str(group))
        used_indices = []
        sentence_num = 0
        for sentence in group:
            # print("original sentence: " + sentence)
            idx = 0
            to_break = False
            # words = sentence.split(separator=None, maxsplit=-1)
            split = sentence.split()
            og_word = split[index]
            split_filler = []
            for word in split:
              split_filler.append(word)
            hyp_split = hypotheses[group_num][sentence_num].split()
            # print("split: " + str(split))
            # print("hyp_split: " + str(hyp_split))
            for index in range(len(split)):
              # word = word.lower()
              # print("word: " + word)
              # print("idx: " + str(idx))
              # print("used_indices: " + str(used_indices))
              if idx not in used_indices:
                    # print("idx is not in used_indices")
                    if split[index].lower() in colors:
                        rand = random.randint(0, len(colors) - 1)
                        while colors[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(colors) - 1)
                        # sentence = sentence[0:sentence.index(word)] + colors[rand] + sentence[sentence.index(word) + len(word):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = colors[rand]
        
                        # if word in hyp_split:
                        #   hyp_index = hyp_split.index(word)
                        # if hyp_index >= 0:
                        #   hyp_split[hyp_index] = word
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in people:
                        rand = random.randint(0, len(people) - 1)
                        while people[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(people) - 1)
                        # print(sentence.index(word))
                        # print(rand)
                        # print(len(sentence))
                        # if sentence.index(split[index]) != 0:
                        #   sentence = sentence[0:sentence.index(split[index])] + people[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        # else:
                        #   sentence = people[rand] + sentence[len(split[index]): len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = people[rand]
                        
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in animals:
                        rand = random.randint(0, len(animals) - 1)
                        while animals[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(animals) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + animals[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = animals[rand]
                        
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in nums:
                        rand = random.randint(0, len(nums) - 1)
                        while nums[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(nums) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + nums[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        split_filler[index] = nums[rand]
                       
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in time:
                        rand = random.randint(0, len(time) - 1)
                        while time[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(time) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + time[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = time[rand]
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in is_not:
                        rand = random.randint(0, len(is_not) - 1)
                        while is_not[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(is_not) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + is_not[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = is_not[rand]
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in are_not:
                        rand = random.randint(0, len(are_not) - 1)
                        while are_not[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(are_not) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + are_not[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = are_not[rand]
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in vehicles:
                        rand = random.randint(0, len(vehicles) - 1)
                        while vehicles[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(vehicles) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + vehicles[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = vehicles[rand]
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in temp:
                        rand = random.randint(0, len(temp) - 1)
                        while temp[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(temp) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + temp[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = temp[rand]
                        used_indices.append(idx)
                        to_break = True
                    elif split[index].lower() in prepositions:
                        rand = random.randint(0, len(prepositions) - 1)
                        while prepositions[rand].lower() == split[index].lower():
                            rand = random.randint(0, len(prepositions) - 1)
                        # sentence = sentence[0:sentence.index(split[index])] + prepositions[rand] + sentence[sentence.index(split[index])+ len(split[index]):len(sentence)]
                        hyp_index = -1
                        if split[index].lower() in hyp_split:
                          hyp_index = hyp_split.index(split[index].lower())
                        if hyp_index >= 0:
                          hyp_split[hyp_index] = split[index].lower()
                        
                        split_filler[index] = prepositions[rand]
                        used_indices.append(idx)
                        to_break = True
                    # print(' '.join(split))
                    # print(' '.join(hyp_split))
              if to_break == True:
                # print("new sentence: " + sentence)
                hyp_index = -1
                if og_word in hyp_split:
                  hyp_index = hyp_split.index(og_word)
                if hyp_index >= 0:
                  hyp_split[hyp_index] = split[index].lower()
                break
              idx += 1
            sentence_num += 1
            perturbed_hyps.append(' '.join(hyp_split))
            perturbations.append(' '.join(split))
        group_num += 1
    # for group in contrast_set:
    #   for sentence in group:
    #     perturbations.append(sentence)
    # print("length: " + str(len(perturbations)))


    # tokenized_examples = tokenizer(
    #     examples['premise'],
    #     examples['hypothesis'],
    #     truncation=True,
    #     max_length=max_seq_length,
    #     padding='max_length'
    # )

    # tokenized_examples = tokenizer(
    #     examples['premise'][:100],
    #     examples['hypothesis'][:100],
    #     truncation=True,
    #     max_length=max_seq_length,
    #     padding='max_length'
    # )

    # print(examples['premise'][:100])
    # print(perturbations)
    # print(contrast_set)

    tokenized_examples = tokenizer(
        perturbations[:num_exs],
        perturbed_hyps[:num_exs],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )
    # print(len(perturbations[:num_exs]))

    tokenized_examples['label'] = examples['label'][:num_exs]
    return tokenized_examples


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli', 'anli',)}
        # default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        # for regular SNLI
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # for contrast sets:
        # prepare_train_dataset = prepare_eval_dataset = \
        #     lambda exs: prepare_contrast_sets(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli', ):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    if dataset_id == ('anli',):
        # remove ANLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    # print(dataset)
    if training_args.do_train:
        ## if snli:
        train_dataset = dataset['train']
        ## if anli:
        # train_dataset = dataset['train_r1']
        # train_dataset = dataset['train_r2']
        # train_dataset = dataset['train_r3']
        ## with train being train_r1, train_r2, or train_r3
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        ## if snli:
        eval_dataset = dataset[eval_split]
        ## if anli:
        # eval_dataset = dataset['test_r1']
        ## with train being test_r1, test_r2, or test_r3
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
