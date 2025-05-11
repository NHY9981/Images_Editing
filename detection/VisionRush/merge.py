from toolkit.chelper import final_model
import torch
import os


# Trained ConvNeXt and RepLKNet paths (for reference)
convnext_path = '/root/autodl-tmp/output/2025-05-10-23-24-17_convnext_to_competition_BinClass/20250510-23-49-EMA-0.pth'
replknet_path = '/root/autodl-tmp/output/2025-05-10-21-41-57_replknet_to_competition_BinClass/20250510-22-27-EMA-0.pth'


model = final_model()
model.convnext.load_state_dict(torch.load(convnext_path, map_location='cpu')['state_dict'], strict=True)
model.replknet.load_state_dict(torch.load(replknet_path, map_location='cpu')['state_dict'], strict=True)

if not os.path.exists('./final_model_csv'):
    os.makedirs('./final_model_csv')

torch.save({'state_dict': model.state_dict()}, './final_model_csv/final_model.pth')
