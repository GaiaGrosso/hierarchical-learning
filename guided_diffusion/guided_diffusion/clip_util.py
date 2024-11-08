import clip
import torch as th


def clip_embedding(visual, x: th.Tensor):
    '''Compute the CLIP embedding of the input token sequence x.

    Args:
        visual (nn.Module): Visual backbone of the CLIP model.
        x (torch.Tensor): Input token sequence x of shape (B, C, 224, 224).

    Returns:
        x (torch.Tensor): Output token sequence x of shape (B, 7, 7, 768).
        class_token (torch.Tensor): Class token of shape (B, 512).
    '''

    with th.no_grad():
        x = x.type(th.cuda.HalfTensor)
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = th.cat([visual.class_embedding.to(x.dtype) + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        class_token = visual.ln_post(x[:, 0, :])

        class_token = class_token @ visual.proj

        x = x[:, 1:, :]
        x = x.reshape(x.shape[0], 7, 7, x.shape[2])

        return x, class_token