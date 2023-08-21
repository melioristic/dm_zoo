import torch
import torch.nn as nn

from dm_zoo.backbone.AFNO.AFNO import AFNONet
from dm_zoo.dff.DenoisingDiffusionProcess import DenoisingDiffusionConditionalProcess

class LFD(nn.Module):
    def __init__(
            self,
            model_config,
            sampler,
            loss_fn
        ):
        super().__init__()
        self.vae = nn.Identity()
        self.forecast = nn.Identity()

        if "vae" in model_config:
            pass
            
        if "forecast" in model_config:
            forecast_model = [key for key in model_config["forecast"]][0]
            if forecast_model == "AFNO":
                self.forecast = AFNONet(
                   model_config.forecast.AFNO,
                    img_size = model_config.image_size,
                    in_channels = model_config.conditioning_channels,
                    out_channels = model_config.generated_channels,           
                    )
                
            if forecast_model == "ViT":
                pass
                #!TODO
                # layers.append(AFNONet(
                #    model_config.AFNO,
                #     img_size = img_size,
                #     in_channels = in_channels,
                #     out_channels = out_channels,                     
                    
                #     ))
        
        self.diffusion = DenoisingDiffusionConditionalProcess(
            model_config.diffusion.denoising_diffusion_process,
            generated_channels= model_config.generated_channels,
            conditioning_channels= model_config.generated_channels,
            loss_fn=loss_fn,
            sampler=sampler,
            )


    def forward(self, x):
        print(x.shape)
        x = self.vae(x)
        print(x.shape)
        x = self.forecast(x)
        print(x.shape)
        x = self.diffusion(x)

        return x