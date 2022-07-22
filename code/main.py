from config import BATCH_SIZE, MAX_LEN, VOCAB_SIZE
from preprocessing import load_dataloaders
from model import BERTLM, ScheduledOptim
from torch.optim import Adam
from torch import nn
import pytorch_lightning
from pytorch_lightning.lite import LightningLite


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

        criterion: nn.Module = nn.NLLLoss(ignore_index=0)
        optimizer = ScheduledOptim(_optimizer)

        print("Training Start")
        model.train()
        for epoch in range(num_epochs):
            avg_loss = 0
            for i, data in enumerate(train_data):
                optimizer.zero_grad()

                nsp_output, mlm_output = model(
                    data["bert_input"], data["segment_label"]
                )

                nsp_loss = criterion(nsp_output, data["is_next"])
                mlm_loss = criterion(
                    mlm_output.reshape(BATCH_SIZE, VOCAB_SIZE, MAX_LEN),
                    data["bert_label"],
                )
                loss = nsp_loss + mlm_loss
                avg_loss += loss.item()
                self.backward(loss)
                optimizer.step_and_update_lr()

                print(
                    f"Epoch: {epoch}.{i}/{len(train_data)} "
                    f"| Loss: {avg_loss / (i+1):.3f} "
                    f"| NSP Loss: {nsp_loss.item():.3f} "
                    f"| MLM Loss {mlm_loss.item():.3f}"
                )


pytorch_lightning.seed_everything(0)
Lite(devices=1, accelerator="gpu", precision="bf16").run(10)
