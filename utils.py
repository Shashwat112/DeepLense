import os
from engine import trainer, exp_used, model_used, train_param as tp
from fpdf import FPDF
import matplotlib.pyplot as plt


def to_pdf_report(met):
    fig = plt.figure()
    plt.plot(met[:,0], label='MSE')
    plt.legend()
    plt.title('Loss plot')
    fig.savefig('loss_plot.png', bbox_inches='tight')
    pdf = FPDF()
    if exp_used == 'super_resolution':
        mse = met[-1,0]
        ssim = met[-1,1]
        psnr = met[-1,2]
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(40, 10, f'Experiment ID:\t\t{trainer.exp_ID}', ln=True)
    pdf.cell(40, 10, f'Name:\t\t{exp_used}', ln=True)
    pdf.cell(40, 10, f'Model Used:\t\t{model_used}', ln=True)
    pdf.cell(40, 10, f'Weight Initializer:\t\t{tp.weight_init}', ln=True)
    pdf.cell(40, 10, f'Bias Initializer:\t\t{tp.bias_init}', ln=True)
    pdf.cell(40, 10, f'Optimizer:\t\t{tp.optimizer}', ln=True)
    pdf.cell(40, 10, f'Learning Rate:\t\t{tp.learning_rate}', ln=True)
    pdf.cell(40, 10, f'Batch Size:\t\t{tp.batch_size}', ln=True)
    pdf.cell(40, 10, f'Epochs:\t\t{int(met[-1,3])}', ln=True)
    if exp_used == 'super_resolution':
        pdf.cell(40, 10, f'MSE Score:\t\t{mse}', ln=True)
        pdf.cell(40, 10, f'SSIM Score:\t\t{ssim}', ln=True)
        pdf.cell(40, 10, f'PSNR Score:\t\t{psnr}', ln=True)
    pdf.image('loss_plot.png')
    pdf.output(f'reports/exp_{trainer.exp_ID}.pdf')
    os.remove('loss_plot.png')