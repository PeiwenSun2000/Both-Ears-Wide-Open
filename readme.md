# Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation

The official repo for Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation

Recently, diffusion models have achieved great success in mono-channel audio generation. However, when it comes to stereo audio generation, the soundscapes often have a complex scene of multiple objects and directions. Controlling stereo audio with spatial contexts remains challenging due to high data costs and unstable generative models. To the best of our knowledge, this work represents the **first attempt** to address these issues. We first construct a large-scale, simulation-based, and GPT-assisted dataset, **BEWO-1M**, with abundant soundscapes and descriptions even including moving and multiple sources. Beyond text modality, we have also acquired a set of images and rationally paired stereo audios through retrieval to advance multimodal generation. Existing audio generation models tend to generate rather random spatial audio. To provide accurate guidance for Latent Diffusion Models, we introduce the **SpatialSonic** model utilizing spatial-aware encoders and azimuth state matrices to reveal reasonable spatial guidance. By leveraging spatial guidance, our unified model not only achieves the objective of generating immersive and controllable spatial audio from text and image but also enables interactive audio generation during inference. Finally, under fair settings, we conduct subjective and objective evaluations on simulated and real-world data to compare our approach with prevailing methods. The results demonstrate the effectiveness of our method, highlighting its capability to generate spatial audio that adheres to physical rules. Our demos are available at https://immersive-audio.github.io/. Our code, model, and dataset will be released soon.

## Open-source schedule

All of our code, dataset, and checkpoints will be released once the paper finishes the review process.
