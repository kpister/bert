import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning
from pytorch_lightning.lite import LightningLite
from model import BERTLM, ScheduledOptim
from config import MAX_LEN, VOCAB_SIZE
from preprocessing import load_dataloaders


def cycle(dl):
    while True:
        for d in dl:
            yield d


class Lite(LightningLite):
    def run(self, num_epochs):
        model = BERTLM()
        _optimizer = Adam(
            model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        model, _optimizer = self.setup(model, _optimizer)
        train_data, val_data, test_data = self.setup_dataloaders(*load_dataloaders())
        train_data_len = len(train_data)
        train_data = cycle(train_data)

        criterion: nn.Module = nn.NLLLoss(ignore_index=0)
        optimizer = ScheduledOptim(_optimizer)

        print("Training Start")
        model.train()
        epoch = 0
        avg_loss = 0
        for i, data in enumerate(train_data):
            if i % train_data_len == 0:
                i = i % train_data_len
                epoch += 1
                if epoch >= num_epochs:
                    break
            optimizer.zero_grad()

            nsp_output, mlm_output = model(data["bert_input"], data["segment_label"])

            nsp_loss = criterion(nsp_output, data["is_next"])
            mlm_loss = criterion(
                mlm_output.reshape(data["bert_label"].size(0), VOCAB_SIZE, MAX_LEN),
                data["bert_label"],
            )
            loss = nsp_loss + mlm_loss
            avg_loss += loss.item()
            self.backward(loss)
            optimizer.step_and_update_lr()

            if i % 10 == 0:
                print(
                    f"Epoch: {epoch}.{i % train_data_len}/{train_data_len} "
                    f"| Loss: {avg_loss / (i+1):.3f} "
                    f"| NSP Loss: {nsp_loss.item():.3f} "
                    f"| MLM Loss {mlm_loss.item():.3f}"
                )

            if i % 1000 == 0:
                torch.save(model, f"models/m_{epoch}_{i}.pth")


pytorch_lightning.seed_everything(0)
Lite(devices=1, accelerator="gpu", precision="bf16").run(10)
