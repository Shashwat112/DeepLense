from engine import trainer
from utils import to_pdf_report
from models import device, ddp
from torch.distributed import init_process_group, destroy_process_group

def default():
    history = trainer.run()
    return history

def main () -> None:
    if ddp and device=='gpu':
        init_process_group(backend='nccl')
    met = default()
    if ddp and device=='gpu':
        destroy_process_group()
    to_pdf_report(met)
    print ('\nExperiment Report generated!\n')
    
if __name__=='__main__':
    main()
