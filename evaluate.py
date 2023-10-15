import constants
import torch
from dataset import Dataset
from model import FastSpeech2Loss
from torch.utils.data import DataLoader
from tools import log, synth_one_sample, to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, logger=None, vocoder=None):
    # Get dataset
    dataset = Dataset("val.txt", sort=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss().to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Calculating Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        # sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=constants.SAMPLING_RATE,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=constants.SAMPLING_RATE,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message
