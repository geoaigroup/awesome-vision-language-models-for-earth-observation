# A curated list of Visual Language Models papers and resources for earth observation (VLM4EO) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/geoaigroup/awesome-vision-language-models-for-earth-observation/)  

This list is created and maintained by [Ali Koteich](https://github.com/alikoteich) and [Hasan Moughnieh](https://geogroup.ai/author/hasan-moughnieh/) from the GEOspatial Artificial Intelligence (GEOAI) research group at the National Center for Remote Sensing - CNRS, Lebanon.  

We encourage you to contribute to this project according to the following [guidelines](https://github.com/sindresorhus/awesome/blob/main/contributing.md).  

---**If you find this repository useful, please consider giving it a ⭐**

**Table Of Contents**
* [Image Captioning](#image-captioning)
* [Text-Image Retrieval](#text-image-retrieval)
* [Visual Grounding](#visual-grounding)
* [Visual Question Answering](#visual-question-answering)
* [VL4EO Datasets](#vision-language-remote-sensing-datasets)

## Image Captioning
| Title | Paper | Code | Year | Venue |
| ----------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------- | ---- | ------------------- |
| RSGPT: A Remote Sensing Vision Language Model and Benchmark | [Paper](https://arxiv.org/abs/2307.15266) | [code](https://github.com/Lavender105/RSGPT) | 2023 | | 
| Multi-Source Interactive Stair Attention for Remote Sensing Image Captioning | [paper](https://www.mdpi.com/2072-4292/15/3/579) | | 2023 | MDPI Remote Sensing []()|
| VLCA: vision-language aligning model with cross-modal attention for bilingual remote sensing image captioning | [paper](https://ieeexplore.ieee.org/document/10066217) | | 2023 | IEEE Journal of Systems Engineering and Electronics |[]()|
| Towards Unsupervised Remote Sensing Image Captioning and Retrieval with Pre-Trained Language Models | [paper](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B10-4.pdf) | | 2023 | Proceedings of the Japanese Association for Natural Language Processing |[]()|
| Captioning Remote Sensing Images Using Transformer Architecture | [paper](https://ieeexplore.ieee.org/document/10067039/) | | 2023 | International Conference on Artificial Intelligence in Information and Communication |
| Progressive Scale-aware Network for Remote sensing Image Change Captioning | [paper](https://arxiv.org/abs/2303.00355) | | 2023 | |
| Change Captioning: A New Paradigm for Multitemporal Remote Sensing Image Analysis | [paper](https://ieeexplore.ieee.org/document/9847254) | | 2022 | IEEE TGRS []()|
| Generating the captions for remote sensing images: A spatial-channel attention based memory-guided transformer approach| [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197622002317) | [code](https://github.com/GauravGajbhiye/SCAMET_RSIC) | 2022 | Engineering Applications of Artificial Intelligence []()|
| Global Visual Feature and Linguistic State Guided Attention for Remote Sensing Image | [paper](https://ieeexplore.ieee.org/document/9632558) | | 2022 | IEEE TGRS []()|
| Recurrent Attention and Semantic Gate for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9515452) | | 2022 | IEEE TGRS []()|
| NWPU-Captions Dataset and MLCA-Net for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9866055) | [code](https://github.com/HaiyanHuang98/NWPU-Captions) | 2022 | IEEE TGRS []()|
| Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset | [paper](https://ieeexplore.ieee.org/document/9934924) | | 2022 | IEEE TGRS []()|
| A Mask-Guided Transformer Network with Topic Token for Remote Sensing Image Captioning | [paper](https://www.mdpi.com/2072-4292/14/12/2939) | | 2022 | MDPI Remote Sensing []()|
| Multiscale Multiinteraction Network for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9720234) | | 2022  | IEEE JSTARS []()|
| Using Neural Encoder-Decoder Models with Continuous Outputs for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9714367) | | 2022 | IEEE Access []()|
| A Joint-Training Two-Stage Method for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9961235) | | 2022 | IEEE TGRS []()|
| Meta captioning: A meta learning based remote sensing image captioning framework | [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000351) | [code](https://github.com/QiaoqiaoYang/MetaCaptioning.) | 2022 | Elsevier PHOTO []()|
| Exploring Transformer and Multilabel Classification for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9855519) | [code](https://gitlab.lrz.de/ai4eo/captioningMultilabel.) | 2022 | IEEE GRSL []()|
| High-Resolution Remote Sensing Image Captioning Based on Structured Attention | [paper](https://ieeexplore.ieee.org/document/9400386) | | 2022 | IEEE TGRS |
| Transforming remote sensing images to textual descriptions | [paper](https://www.sciencedirect.com/science/article/pii/S0303243422000678) | | 2022 | Int J Appl Earth Obs Geoinf  |
| A Joint-Training Two-Stage Method for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9961235/) | | 2022 |  	IEEE TGRS  |
| A Novel SVM-Based Decoder for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9521989) | | 2021 | IEEE TGRS []()| 
| SD-RSIC: Summarization Driven Deep Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9239371) | [code](https://git.tu-berlin.de/rsim/SD-RSIC) | 2021 | IEEE TGRS|
| Truncation Cross Entropy Loss for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9153154) | | 2021 | IEEE TGRS []()|
| Word-Sentence Framework for Remote Sensing Image Captioning | [paper](https://ieeexplore.ieee.org/document/9308980/?denied=) | | 2021 | IEEE TGRS |
| Toward Remote Sensing Image Retrieval Under a Deep Image Captioning Perspective | [paper](https://ieeexplore.ieee.org/document/9154525) | | 2020 | IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing|
| Remote sensing image captioning via Variational Autoencoder and Reinforcement Learning | [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705120302586) | | 2020 | Elservier Knowledge-Based Systems []()|
| A multi-level attention model for remote sensing image captions | [paper](https://www.mdpi.com/2072-4292/12/6/939) | | 2020 | MDPI Remote Sensing []()|
| LAM: Remote sensing image captioning with Label-Attention Mechanism | [paper](https://www.mdpi.com/2072-4292/11/20/2349) | | 2019 | MDPI Remote Sensing []()|
| Exploring Models and Data for Remote Sensing Image Caption Generation | [paper](https://ieeexplore.ieee.org/document/8240966) | | 2017 | IEEE TGRS []()|



## Text-Image Retrieval

| Title | Paper | Code | Year | Venue |
| ----------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------- | ---- | ------------------- |
| An End-to-End Framework Based on Vision-Language Fusion for Remote Sensing Cross-Modal Text-Image Retrieval | [paper](https://www.mdpi.com/2227-7390/11/10/2279) |  | 2023 | MDPI Mathematics |
| Contrasting Dual Transformer Architectures for Multi-Modal Remote Sensing Image Retrieval | [paper](https://www.mdpi.com/2076-3417/13/1/282) | | 2023 | MDPI Applied Sciences|
| Reducing Semantic Confusion: Scene-aware Aggregation Network for Remote Sensing Cross-modal Retrieval | [paper](https://dl.acm.org/doi/abs/10.1145/3591106.3592236) |[code](https://github.com/kinshingpoon/SWAN-pytorch) | 2023 | ICMR'23|
| An Unsupervised Cross-Modal Hashing Method Robust to Noisy Training Image-Text Correspondences in Remote Sensing| [Paper](https://ieeexplore.ieee.org/document/9897500) | [code](https://git.tu-berlin.de/rsim/chnr) | 2022 | IEEE ICIP |
| Unsupervised Contrastive Hashing for Cross-Modal Retrieval in Remote Sensing | [Paper](https://ieeexplore.ieee.org/document/9746251) | [code](https://git.tu-berlin.de/rsim/duch) | 2022 | IEEE ICASSP |
| Multisource Data Reconstruction-Based Deep Unsupervised Hashing for Unisource Remote Sensing Image Retrieval| [Paper](https://ieeexplore.ieee.org/abstract/document/10001754) | [code](https://github.com/sunyuxi/MrHash) | 2022 | IEEE TGRS |
| MCRN: A Multi-source Cross-modal Retrieval Network for remote sensing | [paper](https://www.sciencedirect.com/science/article/pii/S156984322200259X) |[code](https://github.com/xiaoyuan1996/MCRN) | 2022 | Int J Appl Earth Obs Geoinf|
| Knowledge-Aware Cross-Modal Text-Image Retrieval for Remote Sensing Images|[paper](https://ceur-ws.org/Vol-3207/paper4.pdf)||2022||
| Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval | [paper](https://ieeexplore.ieee.org/document/9437331) | | 2022 | IEEE TGRS |
| Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information | [paper](https://ieeexplore.ieee.org/document/9745546) | [code](https://github.com/xiaoyuan1996/GaLR) | 2022 | IEEE TGRS |
| Multilanguage Transformer for Improved Text to Remote Sensing Image Retrieval | [paper](https://ieeexplore.ieee.org/document/9925582) | | 2022 | IEEE JSTARS |
| CLIP-RS: A Cross-modal Remote Sensing Image Retrieval Based on CLIP, a Northern Virginia Case Study | [paper](https://vtechworks.lib.vt.edu/handle/10919/110853) |  | 2022 | Virginia Polytechnic Institute and State University |
| A Lightweight Multi-Scale Crossmodal Text-Image Retrieval Method in Remote Sensing | [paper](https://ieeexplore.ieee.org/document/9594840) | [code](https://github.com/xiaoyuan1996/retrievalSystem) | 2022 | IEEE TGRS |
| Toward Remote Sensing Image Retrieval under a Deep Image Captioning Perspective | [paper](https://ieeexplore.ieee.org/document/9154525)  |  | 2020 | IEEE JSTARS|
| TextRS: Deep bidirectional triplet network for matching text to remote sensing images | [paper](https://www.mdpi.com/2072-4292/12/3/405) |  | 2020 | MDPI Remote Sensing |
| Deep unsupervised embedding for remote sensing image retrieval using textual cues | [paper](https://www.mdpi.com/2076-3417/10/24/8931) |  | 2020 | MDPI Applied Sciences |

## Visual Grounding
| Title | Paper | Code | Year | Venue |
| ----------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------- | ---- | ------------------- |
| LaLGA: Multi-Scale Language-Aware Visual Grounding on Remote Sensing Data |[paper](https://www.researchgate.net/publication/373146282_LaLGA_Multi-Scale_LanguageAware_Visual_Grounding_on_Remote_Sensing_Data)| [code](https://github.com/like413/OPT-RSVG) | 2023 | | 
| Text2Seg: Remote Sensing Image Semantic Segmentation via Text-Guided Visual Foundation Models | [paper](https://arxiv.org/abs/2304.10597) | [code](https://github.com/Douglas2Code/Text2Seg) | 2023    | 
| RSVG: Exploring Data and Models for Visual Grounding on Remote Sensing Data| [paper](https://arxiv.org/abs/2210.12634) | [code](https://github.com/ZhanYang-nwpu/RSVG-pytorch) |  2022| IEEE TGRS | 

## Visual Question Answering
| Title | Paper | Code | Year | Venue |
| ----------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------- | ---- | ------------------- |
| LIT-4-RSVQA: Lightweight Transformer-based Visual Question Answering in Remote Sensing | [paper](https://arxiv.org/abs/2306.00758) | [code](https://git.tu-berlin.de/rsim/lit4rsvqa) | 2023 | IEEE IGARSS |
| A Spatial Hierarchical Reasoning Network for Remote Sensing Visual Question Answering | [paper](https://ieeexplore.ieee.org/document/10018408) | | 2023 | IEEE TGRS |
| Multi-Modal Fusion Transformer for Visual Question Answering in Remote Sensing | [paper](https://arxiv.org/abs/2210.04510) | [code](https://git.tu-berlin.de/rsim/multi-modal-fusion-transformer-for-vqa-in-rs) | 2022 | SPIE Image and Signal Processing for Remote Sensing |
| Change Detection Meets Visual Question Answering | [paper](https://ieeexplore.ieee.org/abstract/document/9901476) | [code](https://github.com/YZHJessica/CDVQA) | 2022 | IEEE TGRS |
| Prompt-RSVQA: Prompting visual context to a language model for Remote Sensing Visual Question Answering | [paper](https://ieeexplore.ieee.org/document/9857471) |  | 2022 | CVPRW |
| From Easy to Hard: Learning Language-guided Curriculum for Visual Question Answering on Remote Sensing Data | [paper](https://ieeexplore.ieee.org/abstract/document/9771224)| [code](https://github.com/YZHJessica/VQA-easy2hard) | 2022 | IEEE TGRS |
| Language Transformers for Remote Sensing Visual Question Answering | [paper](https://ieeexplore.ieee.org/document/9884036) | | 2022 | IEEE IGARSS |
| Bi-Modal Transformer-Based Approach for Visual Question Answering in Remote Sensing Imagery | [paper](https://ieeexplore.ieee.org/document/9832935) | | 2022 | IEEE TGRS | 
| Mutual Attention Inception Network for Remote Sensing Visual Question Answering | [paper](https://ieeexplore.ieee.org/document/9444570) | [code](https://github.com/spectralpublic/RSIVQA) | 2022 | IEEE TGRS |
| RSVQA meets BigEarthNet: a new, large-scale, visual question answering dataset for remote sensing | [paper](https://ieeexplore.ieee.org/document/9553307) | [code](https://github.com/syvlo/RSVQAxBEN) | 2021 | IEEE IGARSS |
| RSVQA: Visual Question Answering for Remote Sensing Data | [paper](https://ieeexplore.ieee.org/abstract/document/9088993) | [code](https://github.com/syvlo/RSVQA) | 2020 | IEEE TGRS |
| How to find a good image-text embedding for remote sensing visual question answering? | [paper](https://arxiv.org/abs/2109.11848) | | 2021 | CEUR Workshop Proceedings |

## Vision-Language Remote Sensing Datasets
| Name | Link | Paper Link | Description |
| --- | --- | --- | --- |
| LAION-EO | [link](https://huggingface.co/datasets/mikonvergence/LAION-EO) | [Paper Link](https://arxiv.org/abs/2309.15535) | Size : 24,933 samples with 40.1% english captions as well as other common languages from LAION-5B <br> mean height of 633.0 pixels (up to 9,999) and mean width of 843.7 pixels (up to 19,687) <br> Platforms : Based on LAION-5B <br> |
| RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model| [Link](https://github.com/om-ai-lab/RS5M) | [Paper Link](https://arxiv.org/abs/2306.11300) | Size: 5 million remote sensing images with English descriptions <br>Resolution : 256 x 256 <br> Platforms:  11 publicly available image-text paired dataset<br>|
| Remote Sensing Visual Question Answering Low Resolution Dataset(RSVQA LR)| [Link](https://zenodo.org/record/6344334) | [Paper Link](https://arxiv.org/abs/2003.07333) | Size: 772 images & 77,232 questions and answers <br>Resolution : 256 x 256 <br> Platforms: Sentinel-2 and Open Street Map<br>Use: Remote Sensing Visual Question Answering <br>|
| Remote Sensing Visual Question Answering High Resolution Dataset(RSVQA HR)| [Link](https://zenodo.org/record/6344367) | [Paper Link](https://arxiv.org/abs/2003.07333) | Size: 10,659 images & 955,664 questions and answers <br>Resolution : 512 x 512  <br> Platforms: USGS and Open Street Map<br>Use: Remote Sensing Visual Question Answering <br>|
| Remote Sensing Visual Question Answering BigEarthNet Dataset (RSVQA x BEN)| [Link](https://zenodo.org/record/5084904) | [Paper Link](https://rsvqa.sylvainlobry.com/IGARSS21.pdf) | Size: 140,758,150 image/question/answer triplets <br>Resolution : High-resolution (15cm)  <br> Platforms: Sentinel-2, BigEarthNet and Open Street Map<br>Use: Remote Sensing Visual Question Answering <br>|
| FloodNet Visual Question Answering Dataset| [Link](https://drive.google.com/drive/folders/1g1r419bWBe4GEF-7si5DqWCjxiC8ErnY?usp=sharing) | [Paper Link](https://arxiv.org/abs/2012.02951) | Size: 11,000 question-image pairs <br>Resolution :  224 x 224  <br> Platforms: UAV-DJI Mavic Pro quadcopters, after Hurricane Harvey<br>Use: Remote Sensing Visual Question Answering <br>|
| Change Detection-Based Visual Question Answering Dataset| [Link](https://github.com/YZHJessica/CDVQA) | [Paper Link](https://ieeexplore.ieee.org/abstract/document/9901476) | Size: 2,968 pairs of multitemporal images and more than 122,000 question–answer pairs <br> Classes: 6 <br> Resolution :  512×512 pixels  <br> Platforms: It is based on semantic change detection dataset (SECOND)<br>Use: Remote Sensing Visual Question Answering <br>|
| Remote Sensing Image Captioning Dataset (RSICap) | [link]( https://github.com/Lavender105/RSGPT) | [Paper Link](https://arxiv.org/abs/2307.15266) |  RSICap comprises 2,585 human-annotated captions with rich and high-quality information <br> This dataset offers detailed descriptions for each image, encompassing scene descriptions (e.g., residential area, airport, or farmland) as well as object information (e.g., color, shape, quantity, absolute position, etc) <br> |
| Remote Sensing Image Captioning Evaluation Dataset (RSIEval)| [link]( https://github.com/Lavender105/RSGPT) | [Paper Link](https://arxiv.org/abs/2307.15266) | 100 human-annotated captions and 936 visual question-answer pairs with rich information and open-ended questions and answers.<br> Can be used for Image Captioning and Visual Question-Answering tasks <br> |
| Revised Remote Sensing Image Captioning Dataset (RSCID)| [Link](https://drive.google.com/open?id=0B1jt7lJDEXy3aE90cG9YSl9ScUk) | [Paper Link](https://arxiv.org/pdf/1712.07835) | Size: 10,921 images with five captions per image <br> Number of Classes: 30 <br>Resolution :  224 x 224  <br> Platforms: Google Earth, Baidu Map, MapABC and  Tianditu<br>Use: Remote Sensing Image Captioning <br>|
|Revised University of California Merced dataset (UCM-Captions)| [Link](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA) | [Paper Link](https://ieeexplore.ieee.org/document/7546397) | Size: 2,100 images with five captions per image <br> Number of Classes: 21 <br>Resolution :  256 x 256  <br> Platforms: USGS National Map Urban Area Imagery collection<br>Use: Remote Sensing Image Captioning <br>|
| Revised Sydney-Captions Dataset| [Link](https://pan.baidu.com/s/1hujEmcG) | [Paper Link](https://ieeexplore.ieee.org/document/7546397) | Size: 613 images with five captions per image <br> Number of Classes: 7 <br>Resolution :  500 x 500<br> Platforms: GoogleEarth<br> Use: Remote Sensing Image Captioning <br>|
| LEVIR-CC dataset| [Link](https://drive.google.com/drive/folders/1cEv-BXISfWjw1RTzL39uBojH7atjLdCG?usp=sharing) | [Paper Link](https://ieeexplore.ieee.org/document/9934924) | Size: 10,077 pairs of RS images and 50,385 corresponding sentences <br> Number of Classes: 10  <br>Resolution :  1024 × 1024 pixels<br> Platforms: Beihang University<br> Use: Remote Sensing Image Captioning <br>|
| NWPU-Captions dataset| [images_Link](https://pan.baidu.com/s/1hmuWwnfPy2eZxxGxt6XuSg), [info_Link](https://github.com/HaiyanHuang98/NWPU-Captions/blob/main/dataset_nwpu.json) | [Paper Link](https://ieeexplore.ieee.org/document/9866055/) | Size: 31,500 images with 157,500 sentences <br> Number of Classes: 45  <br>Resolution : 256 x 256 pixels<br> Platforms: based on NWPU-RESISC45 dataset <br> Use: Remote Sensing Image Captioning <br>|
| Remote sensing Image-Text Match dataset (RSITMD)| [Link](https://drive.google.com/file/d/1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6/view?usp=sharing) | [Paper Link](https://ieeexplore.ieee.org/document/9437331) | Size: 23,715 captions for 4,743 images <br> Number of Classes: 32 <br>Resolution :  500 x 500  <br> Platforms: RSCID and GoogleEarth <br> Use: Remote Sensing Image-Text Retrieval<br>|
| PatterNet| [Link](https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/EYSPYqBztbBBqS27B7uM_mEB3R9maNJze8M1Qg9Q6cnPBQ?e=MSf977) | [Paper Link](https://arxiv.org/abs/1706.03424) | Size: 30,400 images <br> Number of Classes: 38 <br>Resolution :  256 x 256  <br> Platforms: Google Earth imagery and via the Google Map AP <br> Use: Remote Sensing Image Retrieval<br>|
| Dense Labeling Remote Sensing Dataset (DLRSD)| [Link](https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/EVjxkus-aXRGnLFxWA5K440B_k-WNNR5-BT1I6LTojuG7g?e=rgSMHi) | [Paper Link](https://www.mdpi.com/2072-4292/10/6/964) | Size: 2,100 images <br> Number of Classes: 21 <br>Resolution :  256 x 256  <br> Platforms: Extension of the UC Merced  <br> Use: Remote Sensing Image Retrieval (RSIR), Classification and Semantic Segmentation<br>|
| Dior-Remote Sensing Visual Grounding Dataset (RSVGD) | [Link](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_) | [Paper Link](https://ieeexplore.ieee.org/document/10056343) | Size: 38,320 RS image-query pairs and 17,402 RS images<br>Number of Classes: 20<br>Resolution : 800 x 800  <br> Platforms: DIOR dataset  <br> Use: Remote Sensing Visual Grounding <br>|
| Visual Grounding in Remote Sensing Images | [link](https://sunyuxi.github.io/publication/GeoVG) | [Paper Link](https://dl.acm.org/doi/abs/10.1145/3503161.3548316) | Size : 4,239 images including 5,994 object instances and 7,933 referring expressions <br> Images are 1024×1024 pixels<br>Platforms: multiple sensors and platforms (e.g. Google Earth)  <br> |
| Remote Sensing Image Scene Classification (NWPU-RESISC45) | [Link](https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs) | [Paper Link](https://arxiv.org/pdf/1703.00121v1.pdf) | Size: 31,500 images <br>Number of Classes: 45<br>Resolution : 256 x 256 pixels  <br> Platforms: Google Earth  <br> Use: Remote Sensing Image Scene Classification  <br>|

<!--
| High Resolution Remote Sensing Detection (HRRSD) | [Link](https://drive.google.com/open?id=1bffECWdpa0jg2Jnm7V0oCyFFh0N-EIkr) | [Paper Link](https://ieeexplore.ieee.org/document/8676107) | Size: 21,761 images and 55,740 object instances <br>Number of Classes: 13<br>Resolution : spatial resolution from 0.15-m to 1.2-m  <br> Platforms: Google Earth and Baidu Map  <br> Use: Remote Sensing Object Detection <br>|
| Dior Dataset | [Link](https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | [Paper Link](https://arxiv.org/abs/1909.00133) | Size: 23,463 images and 192,518 object instances <br>Number of Classes: 20<br>Resolution : 800 x 800  <br> Platforms: Technical University of Munich · Northwestern Polytechnical University · Zhengzhou Institute of Surveying and Mapping  <br> Use: Remote Sensing Object Detection <br>|
| Remote Sensing Object Detection (RSOD) |Each object has its own link: [aircraft](http://pan.baidu.com/s/1eRWFV5C), [playground](http://pan.baidu.com/s/1nuD4KLb), [overpass](http://pan.baidu.com/s/1kVKAFB5) and [oiltank](http://pan.baidu.com/s/1kUZn4zX) | [Paper Link](http://ieeexplore.ieee.org/abstract/document/7827088/) | Size: 976 images and 6,950 object instances<br>Number of Classes: 4<br>Resolution : range from 0.3m to 3m  <br> Platforms: Google Earth and Tianditu  <br> Use: Remote Sensing Object Detection <br>|
| DOTA-v1.0 | [Training_Set](https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK?usp=sharing), [Validation_Set](https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing), and [Testing_set](https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl?usp=sharing) | [Paper Link](https://arxiv.org/abs/1711.10398) | Size: 2,806 images and 188, 282 instances<br>Number of Classes: 15<br>Resolution : range from 800 × 800 to 20,000 × 20,000 pixels  <br> Platforms: Google Earth, GF-2 and JL-1 satellite provided by the China Centre for Resources Satellite Data and Application, and aerial images provided by CycloMedia B.V  <br> Use: object detection in aerial images <br>|
| DOTA-v1.5 | [Training_Set](https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK?usp=sharing), [Validation_Set](https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing), and [Testing_set](https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl?usp=sharing) | [Paper Link](https://arxiv.org/abs/1711.10398) | Size: 2,806 images with 403,318 instances in total<br>Number of Classes: 16<br>Resolution : range from 800 × 800 to 20,000 × 20,000 pixels  <br> *uses the same images as DOTA-v1.0, but the extremely small instances (less than 10 pixels) are also annotated. Moreover, a new category, ”container crane” is added.  <br> Use: object detection in aerial images <br>|
| DOTA-v2.0 |You need to download DOTA-v1.0 images, and then download the extra images and annotations of [DOTA-v2.0](https://whueducn-my.sharepoint.com/:f:/g/personal/2014301200247_whu_edu_cn/EiJ3JsfWPqhPn2955rjdtxoBZUFYWCX2ZXOtbZ-GT0I7Qw?e=XjeBMB) | [Paper Link](https://arxiv.org/abs/1711.10398) | Size: 11,268 images and 1,793,658 instances<br>Number of Classes: 18<br>Resolution : range from 800 × 800 to 20,000 × 20,000 pixels  <br> *Compared to DOTA-v1.5, it further adds the new categories of ”airport” and ”helipad”.  <br> Use: object detection in aerial images <br>|
| iSAID Dataset | [Training_Set](https://drive.google.com/drive/folders/19RPVhC0dWpLF9Y_DYjxjUrwLbKUBQZ2K?usp=sharing), [Validation_Set](https://drive.google.com/drive/folders/17MErPhWQrwr92Ca1Maf4mwiarPS5rcWM?usp=sharing), [Testing_Set](https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl?usp=sharing), and [testing_images_info](https://drive.google.com/open?id=1nQokIxSy3DEHImJribSCODTRkWlPJLE3) | [Paper Link](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf) | Size: 2,806 images with 655,451 object instances<br>Number of Classes: 15<br>Resolution : high resolution  <br> Platforms: Dota Dataset  <br> Use: semantic segmentation or object detection <br>|
| WHU dataset |[link](https://www.kaggle.com/datasets/xiaoqian970429/whu-building-dataset) - http://gpcv.whu.edu.cn/data/building_dataset.html | [Paper Link](https://arxiv.org/pdf/2208.00657v1.pdf) | Size: more than 220, 000 independent buildings <br>Number of Classes: 1<br>Resolution : 0.075 m spatial resolution and 450 km2 covering in Christchurch, New Zealand  <br> Platforms: QuickBird, Worldview series, IKONOS, ZY-3 and  6 neighboring satellite images covering 550 km2 on East Asia with 2.7 m ground resolution.<br> Use: Remote Sensing Building detection and change detection <br>|
| Vaihingen/Enz, Germany dataset |[link](https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/) | [Paper Link](https://arxiv.org/pdf/2206.09731v2.pdf) | Size: The data set contains 33 patches (of different sizes), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic <br>Number of Classes:  five foreground classes and one background class <br>Resolution : 9 cm resolution <br> Platforms:  Intergraph/ZI DMC block, Leica ALS50 system and digital aerial cameras carried out by the German Association of Photogrammetry and Remote Sensing (DGPF) <br> Use: Urban Classification, 3D Building Reconstruction and Semantic Labeling <br>|
| Potsdam dataset |[link](https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/) | [Paper Link](https://arxiv.org/pdf/2206.09731v2.pdf) | Size: 38 patches (of the same size), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic <br>Number of Classes: same category information as the Vaihingen dataset<br>Resolution : 6000x6000 pixels and 5cm resolution <br> Platforms:  Google Maps and OSM (DGPF)<br> Use: Semantic Segmentation <br>|
-->

## Related Repositories 
- [awesome-RSVLM](https://github.com/om-ai-lab/awesome-RSVLM)
- [awesome-remote-sensing-vision-language-models](https://github.com/lzw-lzw/awesome-remote-sensing-vision-language-models)
- [awesome-remote-image-captioning](https://github.com/iOPENCap/awesome-remote-image-captioning)
 <!-- 
- [awesome-satellite-imagery-datasets][https://github.com/chrieke/awesome-satellite-imagery-datasets]
-->

---**Stay tuned for continuous updates and improvements! 🚀**



