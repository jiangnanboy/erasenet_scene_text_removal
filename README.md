## erasenet_scene_text_removal

This repository is from [EraseNet](https://github.com/lcy0604/EraseNet). 

For this project, I made some adaptations and changes:

1. During the training process, evaluation was added
 
2. A single picture inference interface was added
 
3. The pytorch version was upgraded

4. Distributed training was removed

5. Some errors in training were changed

### Environment

```
python = 3.9
pytorch = 1.12.1
torchvision = 0.13.0
```

### Training

Once the data is well prepared, you can begin training:
```
python train.py --batchSize 16 \
  --dataRoot 'your path' \
  --modelsSavePath 'your path' \
  --logPath 'your path'  \
```

### infer

If you want to infer the results, run [infer.py]

```
    args = init_args()
    print('args: {}'.format(args))
    model = init_model(args)
    print('model: {}'.format(model))
    img_path = './example/all_images/118.jpg'
    img = Image.open(img_path)
    img_trains = ImageTransform(args.loadSize)
    img = img_trains(img.convert('RGB'))
    with torch.no_grad():
        start = time.time()
        img = img.to(DEVICE)
        img = img.unsqueeze(0)
        out1, out2, out3, g_imgs, mm = model(img)
        g_imge = g_imgs.data.cpu()
        save_image(g_imge, args.savePath + '/result.jpg')
```
