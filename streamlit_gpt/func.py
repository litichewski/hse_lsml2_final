from tqdm import trange
import torch
import torch.nn.functional as F
import regex as re


def generate(
    model,
    tokenizer,
    prompt,
    entry_length=80,
    top_p=0.7,
    temperature=0.85,
):

    model.eval()
    generated_list = []
    filter_value = -float("Inf")

    with torch.no_grad():
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        for i in range(entry_length):
            outputs = model(generated, labels=generated)
            loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token in tokenizer.encode(""):
                break

        output_list = list(generated.squeeze().numpy())
        output_text = tokenizer.decode(output_list)
        generated_list.append(output_text)

    return generated_list[0] 

def create_text(prompt, model, tokenizer):
    generated_text = generate(model.to('cpu'), tokenizer, prompt)
    text = re.sub(r'^[^\w]+', '', generated_text)
    last_dot = text.rfind('.')
    return text[:last_dot+1] if last_dot != -1 else text

