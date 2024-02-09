from micromind.networks.yolo import YOLOv8
from micromind.utils.yolo import get_variant_multiples
import torch

if __name__ == "__main__":

    conf = "n"
    w, r, d = get_variant_multiples(conf)
    model = YOLOv8(w, r, d, num_classes=80)
    model_dict_keys = list(model.state_dict().keys())
    model_dict = model.state_dict()

    # downloaded_dict = torch.load(download_filename)["model"].state_dict()
    # downloaded_dict_keys = downloaded_dict.keys()


    downloaded_dict = torch.load(f"../../../ultralytics/modelll.pt") # scaricato
    downloaded_dict_keys = downloaded_dict.keys()

    for i, key in enumerate(downloaded_dict_keys):
        assert model_dict[model_dict_keys[i]].shape == downloaded_dict[key].shape
        if downloaded_dict[key].size != 1:
            tmp = torch.tensor(downloaded_dict[key], dtype=downloaded_dict[key].dtype)
        else:
            tmp = torch.tensor([downloaded_dict[key]], dtype=downloaded_dict[key].dtype)
        model_dict[model_dict_keys[i]] = tmp
        assert (
            model_dict[model_dict_keys[i]] == downloaded_dict[key]
        ).all(), f"Failed at: {model_dict_keys[i]}."
    torch.save(model_dict, "test.pt")
    
    breakpoint()
    model.load_state_dict(dict)
    breakpoint()