import torch
with open("/scratch/s3668320/AML/test.txt", "w") as f:
    f.write(str(torch.cuda.is_available()))
