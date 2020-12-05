import torch

def evaluate_bleu(model, iterator, criterion):

    model.eval()

    bleu_scores = []
    trg_list = []
    output_list = []
    src_list = []
    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            for b in range(src.size(1)):
                topv, topi = output.data.topk(1) #greedy
                str_src = [SRC.vocab.itos[idx] for idx in src[1:,b]]
                str_out = [TRG.vocab.itos[idx] for idx in topi[1:,b]]
                str_trg = [TRG.vocab.itos[idx] for idx in trg[1:,b]]

                src_list.append(" ".join(str_src))
                output_list.append(" ".join(str_out))
                trg_list.append(" ".join(str_trg))

                bleu_scores.append(sentence_bleu([str_trg], str_out, smoothing_function=SmoothingFunction().method4))

        data = {"bleu_score": bleu_scores,
                "src":src_list,
                "trg":trg_list,
                "translation": output_list}

        df = pd.DataFrame(data)
        df.to_csv("seq2seq_test.csv")
    return sum(bleu_scores) / len(bleu_scores)


def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
