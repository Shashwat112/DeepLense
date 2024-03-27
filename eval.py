from engine import evaluator, model_used
import os

checkpoint = input("Model checkpoint to use for evaluation (type -1 for latest checkpoint): ")
if checkpoint == '-1':
    checkpoint = max(os.listdir(f'checkpoints/{model_used}'), key=lambda x:int(x[-4]))

evaluator.run(checkpoint)