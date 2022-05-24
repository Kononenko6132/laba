Лабораторная работа 2. Переход между цветовыми пространствами. Линейный и нелинейный переход. Мера цветовой разницы. Функции преобразования яркости. Гамма, логарифмическое, экспоненциаяльное кодирование.

Отобразить изображение по каналам RGB (каждый канал представить как градации серого).


Оригинал                                |  
:--------------------------------------:|
 <img src="original.jpg" width="300"/>|  

```
public void channels(BufferedImage img) throws IOException {
    int h = img.getHeight();
    int w = img.getWidth();
    BufferedImage chR = new BufferedImage(w, h, TYPE_INT_RGB);
    BufferedImage chG = new BufferedImage(w, h, TYPE_INT_RGB);
    BufferedImage chB = new BufferedImage(w, h, TYPE_INT_RGB);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int rgb = img.getRGB(x, y);
            int red = red(rgb);
            int green = green(rgb);
            int blue = blue(rgb);
            chR.setRGB(x, y, rgb(red, red, red));
            chG.setRGB(x, y, rgb(green, green, green));
            chB.setRGB(x, y, rgb(blue, blue, blue));
        }
    }
    save(chR, "result/channels", "r", FORMAT);
    save(chG, "result/channels", "g", FORMAT);
    save(chB, "result/channels", "b", FORMAT);
}
```


Отображение по каналам R                  | Отображение по каналам G                 | Отображение по каналам B                       | 
:----------------------------------------:|:----------------------------------------:|:----------------------------------------------:|
 <img src="channel1.jpg" width="700"/>    |  <img src="channel2.jpg" width="700"/>   |  <img src="channel3.jpg" width="700"/>         |
 
 Лианеризовать изображение обратным гамма преобразованием.
 
 ```
public BufferedImage gammaCorrection(BufferedImage img, double gamma) throws IOException {
    int h = img.getHeight();
    int w = img.getWidth();
    BufferedImage result = new BufferedImage(w, h, TYPE_INT_RGB);
    int[] gammaLUT = new int[256];
    for (int i = 0; i < gammaLUT.length; i++) {
        gammaLUT[i] = (int) (255 * (Math.pow(i / 255f, 1 / gamma)));
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int rgb = img.getRGB(x, y);
            int red = gammaLUT[red(rgb)];
            int green = gammaLUT[green(rgb)];
            int blue = gammaLUT[blue(rgb)];
            result.setRGB(x, y, rgb(red, green, blue));
        }
    }
    save(result, "result/gammaCor", "result", FORMAT);
    return result;
}
```

Отображение по каналам R                  | 
:----------------------------------------:|
 <img src="gamma.jpg" width="300"/>    |  
