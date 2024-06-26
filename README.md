This repository is based on:

https://github.com/AI-secure/semantic-randomized-smoothing

https://github.com/microsoft/denoised-smoothing

https://github.com/Jayfeather1024/DensePure

https://github.com/Ping-C/CertifiedObjectDetection


Run RS and DRS on adv images: 

`cd ccrf`

`python3 calc_rs_{metric_name}.py`


Run RS and DRS on clear images: 

`cd ccrf`

`python3 calc_clear_rs_{metric_name}.py`




Run MS and DMS on adv images: 

`cd ccrf`

`python3 calc_nms_{metric_name}.py`


Run MS and DMS on clear images: 

`cd ccrf`

`python3 calc_clear_nms_{metric_name}.py`




Run DDRS on adv images: 

`cd ccrf`

`python3 calc_ddrs_{metric_name}.py`


Run DDRS on clear images: 

`cd ccrf`

`python3 calc_clear_ddrs_{metric_name}.py`




Run DP on adv images: 

`cd DensePure-master`

`python3 calc_dp_{metric_name}.py`


Run DP on clear images: 

`cd DensePure-master`

`python3 calc_clear_dp_{metric_name}.py`


Used weights (upload to root folder):

* [p1q2.pth](https://drive.google.com/file/d/1fwP7rhwdWwUtV92kkdLx88SXq3SvxfRs/view?usp=sharing)
* [model_best_kadid_0.pkl](https://drive.google.com/file/d/1SZlCQQvV04z8p5kZVWB_eLQFZGzawJHK/view?usp=sharing)
* [metaiqa.pth](https://drive.google.com/file/d/19U_gHdK3Ld9uI23MABC14Pin9Vfw8AQM/view?usp=sharing)
* [KonCept512.pth](https://drive.google.com/file/d/1fzFjWMQw4x_6lv3ZOxaqx3qI6I-q3Lrk/view?usp=sharing)
* [inceptionresnetv2-520b38e4.pth](https://drive.google.com/file/d/1M9looiiWsi_FnWE3R-9N4_zsyhgvzrBp/view?usp=sharing)
* [BL_release.pt](https://drive.google.com/file/d/1mR6c8UfFaryZNXX-Wy1d0NHDOJl5vQX_/view?usp=sharing)
* [256x256_diffusion_uncond.pt](https://drive.google.com/file/d/1QrccFT48o2hEYsdpfLfsfPkNmIlYvvDz/view?usp=sharing)
