# Exposure AI Academy — canlı eğitim animasyonları

Proje 1 (Rock Paper Scissors Vision) için eğitim sırasında oynayan canlı paneller. Projeyi takip eden model
panelleri **baştan yazmaz**: `animations/` klasöründeki iki dosyayı değiştirmeden projeye kopyalar ve eğitim
döngüsüne bağlar. Böylece her öğrencide aynı animasyon çıkar.

| Dosya | Ne yapar |
|---|---|
| `animations/live.py` | `CNNLive` (Model A) ve `FineTuneLive` (Model B). Eğitim sürecinin içinde küçük bir sunucu açar (127.0.0.1, port 8002-8019), modelin o anki ağırlıklarından en çok 3 sn'de bir değer alır. |
| `animations/live_cnn.html` | Tarayıcı sayfası, iki model için aynı. `live.py` ile aynı klasörde durmalı. |

Gereken paketler: `torch`, `numpy`, `Pillow` (Model B için `timm`). Sunucu Python'un kendi `http.server`'ı.

## Kurulum

İki dosyayı notebook'un yanına indir:

```bash
curl -LO https://raw.githubusercontent.com/Darkosxl/exposure-academy-animations/main/animations/live.py
curl -LO https://raw.githubusercontent.com/Darkosxl/exposure-academy-animations/main/animations/live_cnn.html
```

## Modellerin şekli (panel bunlara bakar)

- **Model A:** `nn.Sequential`, ilk 5 elemanı blok: `nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))`, kanallar 32-64-128-128-256; sonra Flatten → Dropout → Linear(12544 → 2000) → ReLU → Dropout → Linear(2000 → 3).
- **Model B:** `timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=3)`. Panel `model.stages` ve `model.head`'i kullanır; hangi kısmın donuk olduğunu `requires_grad`'dan okur (donuk = gri, öğrenen = mavi).

## Eğitim döngüsüne bağlama

`ds` bir `torchvision.datasets.ImageFolder` (panel `ds.targets` ile her sınıftan bir örnek alır), `to_rgb`
normalize edilmiş tensörü `(H, W, 3)` 0..1 görüntüye çevirir, `history` her epoch sonunda bir eleman eklenen liste
(panel epoch sayacını `len(history)`'den okur).

```python
from live import CNNLive, FineTuneLive

def to_rgb(t):  # normalizasyonu geri al
    return (t.permute(1, 2, 0) * torch.tensor(STD) + torch.tensor(MEAN)).clamp(0, 1)

def train(model, epochs, lr, history, name, view=None):
    ...
    for _ in range(epochs):
        for i, (x, y) in enumerate(train_dl, start=1):
            ...  # forward, loss.backward(), opt.step()
            if view: view(model, history, i, len(train_dl), x=x, y=y)        # her batch'ten sonra
        history.append(...)                                                  # epoch sonu metrikleri
        if view: view(model, history, len(train_dl), len(train_dl), force=True)
    return history

# Model A
view_a = CNNLive(model_a, val_ds, to_rgb, CLASSES, epochs=60)
hist_a = train(model_a, 60, 1e-3, [], "model_a", view=view_a)
view_a.close()

# Model B: epochs = iki adımın toplamı
view_b = FineTuneLive(model_b, val_ds, to_rgb, CLASSES, epochs=15)
set_trainable(model_b, [model_b.head])                        # 1) gövde donuk (requires_grad=False)
hist_b = train(model_b, 5, 1e-3, [], "model_b", view=view_b)
set_trainable(model_b, [model_b.stages[-1], model_b.head])    # 2) son aşama çözülür
hist_b = train(model_b, 10, 1e-4, hist_b, "model_b", view=view_b)
view_b.close()
```

Panel eğitim başlayınca tarayıcıda açılır (`webbrowser.open`; VS Code uzak bağlantıda senin bilgisayarında).
Açılmazsa hücre çıktısındaki `Canlı panel: http://localhost:8002/` linkine tıkla.

## Panel ne gösterir

İki panel de aynı 3D kule şehri. Her kule fotoğrafın bir bölgesi; yüksekliği o bölgenin, takip edilen damgaya
(hep aynı kanal) ne kadar benzediği. Sayfa bir turu ~16-17 sn'de oynatır, her tur en yeni değerlerle başlar.
Emoji yok, sayı sadece epoch sayacı.

**Model A** (sıfırdan CNN), 5 blok, blok 1 yavaş, 2-5 hızlı:

1. **Conv 3×3:** gerçek 3×3 damga fotoğrafın üstünden geçer, kuleler yükselir.
2. **BatchNorm:** şehir kayar, su çizgisi ortalamaya gelir.
3. **ReLU:** suyun altında kalan kuleler silinir.
4. **MaxPool:** her 2×2'nin en uzunu kalır, şehir küçülür.

Sonunda şehir tek kuleye iner, Taş / Kağıt / Makas çubukları dolar.

**Model B** (ConvNeXt-Tiny, hazır beyin), Model A ile aynı şehir, farkları gösterir:

1. **Patchify:** fotoğraf 4×4 piksellik, üst üste binmeyen karelere bölünür (çizimde 16×16, gerçekte 56×56); şehir bir anda çıkar.
2. Her aşamadan bir blok, aşama 1 yavaş, 2-4 hızlı:
   - **Conv 7×7:** daha büyük, gerçek 7×7 damga geçer.
   - **LayerNorm:** su çizgisi ortalamaya gelir.
   - **GELU:** suyun altı silinmez, kısa kütüklere ezilir.
   - **Residual:** eski şehir altta açık renkte geri gelir, yeni kısım üstüne eklenir.
3. **Downsample** (aşamalar arasında): her 2×2 karışıp 1 kule olur, şehir küçülür.
4. Donuk aşamalar **gri**, öğrenenler **mavi**. 1. adımda sadece karar kısmı mavi, 2. adımda son aşama da maviye döner. Alttaki nokta sırası (3 / 3 / 9 / 3 blok + karar) aynı renkleri gösterir.
5. Sonunda şehir karar kısmının renginde tek kuleye iner, Taş / Kağıt / Makas çubukları.

Yaklaşıklıklar: her aşamanın sadece ilk bloğu gösterilir; GELU gerçek blokta 1×1 karıştırmadan sonra gelir, panelde
LayerNorm çıktısına uygulanır; kule ızgarası gerçek çözünürlükten küçüktür.
