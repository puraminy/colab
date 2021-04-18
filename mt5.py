import os
import functools

#import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_datasets as tfds
import multilingual_t5.tasks

import t5
import t5.data.utils as ut
from pathlib import Path
DEFAULT_SPM_PATH = "/drive3/pouramini/pretrained/mt5/sentencepiece.model"

DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}

DO_TRAIN = True
DO_EVAL = False
DO_PREDICT = True
DO_EXPORT = True

EXP = "natural_all_atomic"
TRAIN_FILE = EXP + "_train.tsv"
DEV_FILE = "natural_all_atomic_dev.tsv"

FINETUNE_STEPS = 100000 #@param {type: "integer"}
CHECK_POINT_STEP = 5000

MODEL_SIZE = "small" #@param["small", "base", "large", "3B", "11B"]
MODEL_TYPE = "mt5"
MODEL_NAME = MODEL_SIZE + "_" + EXP
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = "/drive3/pouramini"
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, "pretrained", MODEL_TYPE, MODEL_SIZE)
MODELS_DIR = os.path.join(BASE_PRETRAINED_DIR, "models", MODEL_TYPE)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

DATA_DIR = os.path.join(BASE_PRETRAINED_DIR, "data", "atomic")
atomic_tsv_path = {
    "train": os.path.join(DATA_DIR, TRAIN_FILE),
    "validation": os.path.join(DATA_DIR, DEV_FILE)
}

VALIDATION_DIR = os.path.join(MODEL_DIR, DEV_FILE)

tf.io.gfile.makedirs(MODEL_DIR)
os.environ['NO_GCE_CHECK'] = 'true'
model_parallelism, train_batch_size, eval_batch_size, keep_checkpoint_max = {
    "small": (1, 8, 64, 5),
    "base": (1, 8, 48, 4),
    "large": (1, 4, 48, 3),
    "3B": (8, 16, 48, 1),
    "11B": (8, 16, 48, 1)}[MODEL_SIZE]

def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

def atomic_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(atomic_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["input_text", "target_text"], ex)))
  return ds

print("A few raw validation examples...")
for ex in tfds.as_numpy(atomic_dataset_fn("validation").take(5)):
  print(ex)

def trivia_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["atomic: ", normalize_text(ex["input_text"])]),
        "targets": normalize_text(ex["target_text"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
t5.data.TaskRegistry.remove(EXP)
t5.data.TaskRegistry.add(
    EXP,
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=atomic_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[trivia_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
	output_features=DEFAULT_OUTPUT_FEATURES,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    #num_input_examples=num_atomic_examples
)
# Alternative
#t5.data.TaskRegistry.add(
#    EXP,
#    t5.data.TextLineTask,
#    split_to_filepattern=atomic_tsv_path,
#    text_preprocessor=[
#      functools.partial(
#          t5.data.preprocessors.parse_tsv,
#          field_names=["question", "answer"]),
#      trivia_preprocessor
#    ],
#    postprocess_fn=t5.data.postprocessors.lower_text, 
#    metric_fns=[t5.evaluation.metrics.accuracy],
#    num_input_examples=num_atomic_examples
#)
atomic_task = t5.data.TaskRegistry.get(EXP)
ds = atomic_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(5)):
  print(ex)
# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
# The models from our paper are based on the Mesh Tensorflow Transformer.

model = t5.models.MtfModel(
    tpu=None,
    model_dir=MODEL_DIR,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 128, "targets": 32},
    mesh_shape = 'model:1,batch:1',
    mesh_devices = ['gpu:0'],
    learning_rate_schedule=0.003,
    save_checkpoints_steps=CHECK_POINT_STEP,
    keep_checkpoint_max=None, #keep_checkpoint_max if ON_CLOUD else None,
    iterations_per_loop=100,
)

if DO_TRAIN:
    model.finetune(
        mixture_or_task_name=EXP,
        pretrained_model_dir=PRETRAINED_DIR,
        finetune_steps=FINETUNE_STEPS
    )

#tf.disable_eager_execution()
#tf.compat.v1.disable_v2_behavior()
if DO_EVAL:
    tf.io.gfile.makedirs(VALIDATION_DIR)
    print("start evaluation")
    model.batch_size = eval_batch_size #train_batch_size * 4
    model.eval(
        mixture_or_task_name=EXP,
        summary_dir= VALIDATION_DIR,
        checkpoint_steps=-1 #
    )

import random
def print_random_predictions(task_name, n=10):
  """Print n predictions from the validation split of a task."""
  # Grab the dataset for this task.
  ds = t5.data.TaskRegistry.get(task_name).get_dataset(
      split="validation",
      sequence_length={"inputs": 128, "targets": 32},
      shuffle=False)

  def _prediction_file_to_ckpt(path):
    """Extract the global step from a prediction filename."""
    return int(path.split("_")[-2])

  # Grab the paths of all logged predictions.
  prediction_files = tf.io.gfile.glob(
      os.path.join(
          MODEL_DIR,
          VALIDATION_DIR + "/%s_*_predictions" % task_name))
  # Get most recent prediction file by sorting by their step.
  latest_prediction_file = sorted(
      prediction_files, key=_prediction_file_to_ckpt)[-1]

  # Collect (inputs, targets, prediction) from the dataset and predictions file
  results = []
  with tf.io.gfile.GFile(latest_prediction_file) as preds:
    for ex, pred in zip(tfds.as_numpy(ds), preds):
      results.append((tf.compat.as_text(ex["inputs_pretokenized"]),
                      tf.compat.as_text(ex["targets_pretokenized"]),
                      pred.strip()))

  print("<== Random predictions for %s using checkpoint %s ==>\n" %
        (task_name, 
         _prediction_file_to_ckpt(latest_prediction_file)))

  for inp, tgt, pred in random.choices(results, k=10):
    print("Input:", inp)
    print("Target:", tgt)
    print("Prediction:", pred)
    print("Counted as Correct?", tgt == pred)
    print()

if DO_PREDICT:
    print_random_predictions(EXP)

# First import the t5 mixtures so we can load the training mixture's vocabulary.
    import t5.data.mixtures

    question_1 = "علی کتاب خرید علی این کار را برای"
    question_2 = "علی به رضا کمک کرد. رضا "
    question_3 = "علی همه را قانع کرد. در نتیجه دیگران"
    question_4 = "PersonX convinces every ___. As a result, others feel" # persuaded

    questions = [question_1, question_2, question_3, question_4]

    import time
    now = time.time()
    # Write out the supplied questions to text files.
    predict_inputs_path = os.path.join(MODEL_DIR, "predict_inputs_%d.txt" % now)
    predict_outputs_path = os.path.join(MODEL_DIR, "predict_outputs_%d.txt" % now)
    # Manually apply preprocessing by prepending "triviaqa question:".
    with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
      for q in questions:
        f.write("atomic: %s\n" % q.lower())

    model.batch_size = 32  # Min size for small model on v2-8 with parallelism 1.
    model.predict(
      input_file=predict_inputs_path,
      output_file=predict_outputs_path,
      # Select the most probable output token at each step.
      vocabulary=DEFAULT_VOCAB,
      temperature=0,
    )

    # The output filename will have the checkpoint appended so we glob to get 
    # the latest.
    prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
    print("\nPredictions using checkpoint %s:\n" % prediction_files[-1].split("-")[-1])
    with tf.io.gfile.GFile(prediction_files[-1]) as f:
      for q, a in zip(questions, f):
        if q:
          print("Q: " + q)
          print("A: " + a)
          print()

if DO_EXPORT:
    export_dir = os.path.join(MODEL_DIR, "export")

    model.batch_size = 1 # make one prediction per call
    saved_model_path = model.export(
        export_dir,
        checkpoint_step=-1,  # use most recent
        beam_size=1,  # no beam search
        vocabulary=DEFAULT_VOCAB,
        temperature=1.0,  # sample according to predicted distribution
    )
    print("Model saved to:", saved_model_path)

    import tensorflow as tf
    import tensorflow_text  # Required to run exported model.

    def load_predict_fn(model_path):
      if tf.executing_eagerly():
        print("Loading SavedModel in eager mode.")
        imported = tf.saved_model.load(model_path, ["serve"])
        return lambda x: imported.signatures['serving_default'](tf.constant(x))['outputs'].numpy()
      else:
        print("Loading SavedModel in tf 1.x graph mode.")
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        meta_graph_def = tf.compat.v1.saved_model.load(sess, ["serve"], model_path)
        signature_def = meta_graph_def.signature_def["serving_default"]
        return lambda x: sess.run(
            fetches=signature_def.outputs["outputs"].name, 
            feed_dict={signature_def.inputs["input"].name: x}
        )

    predict_fn = load_predict_fn(saved_model_path)

    def answer(question):
      return predict_fn([question])[0].decode('utf-8')

    questions = [ 
        "PersonX brings ___ to the people. As a result, others feel", 
        "PersonX brings ___ to the people. PersonX did this to",
        "PersonX brings ___ to the people. PersonX will be",
        "PersonX brings ___ to the people. As a result, others feel",
        ]

    for question in questions:
        print(answer(question))
