from transformers import AutoModel, AutoTokenizer
import torch


class SimCSE(torch.nn.Module):
    def __init__(self):
        super(SimCSE, self).__init__()
        self.model = AutoModel.from_pretrained("Seznam/simcse-small-e-czech")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        emb = model_output.last_hidden_state[:, 0]
        return emb


def load_simcse():
    tokenizer = AutoTokenizer.from_pretrained("Seznam/simcse-small-e-czech")
    retriever = SimCSE()
    return retriever, tokenizer, "Seznam/simcse-small-e-czech"
