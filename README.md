# Time-Series-Theory

# Trend - tính xu hướng
Trend thể hiện xu hướng thay đổi của dữ liệu theo thời gian, ví dụ giá nhà tăng theo từng năm...

Tính chu kỳ: tính chất lặp lại của dữ liệu theo thời gian.

Tính thời vụ

Nhiễu: thể hiện tính chất không dự đoán trước được của dữ liệu

# Moving average and diferencing
Một cách dự đoán đơn giản là tính `moving average`. Ví dụ gia trị tiếp theo bằng trung bình của 30 ngày trước đó. Cách này loại bỏ rất nhiều nhiễu, nó không dự đoán được xu hướng và tính thời vụ. 

Một cách để tránh được điều này là loại bỏ `trend` và `seasonality` khỏi time series với kỹ thuật `differecing`. Chúng ta sẽ nghiên cứu sự khác nhau giữa giá trị tại thời điểm `t` và giá trị tại thời điểm trước đó (tùy thuộc trường hợp là ngày, tháng, năm...), khi đó chúng ta không có trend và cả seasonality. Sau đó sử dụng `moving average` để dự đoán giá trị cho time series này. Nên nhớ đây là dự đoán cho difference, muốn lấy giá trị thực tế cần cộng với giá trị tại thời điểm trước đó. Cuối cùng ta nhận được đường dự đoán có nhiều noise (noise này xuất hiện từ giá trị tại thời điểm trước đó từ time series ban đầu). Chúng ta có thể loại bỏ noise này bằng cách áp dụng `moving average` cho giá trị tại thời điểm trước.

# Week 2
Áp dụng một số kỹ thuật ML để dự đoán dữ liệu.
## Preparing features and labels
`Features` - một số giá trị liên tuc trong series (chúng ta gọi cái này là `window_size`, nhìn cũng như cái cửa số vậy) và `label` là giá trị tiếp theo.

Ví dụ chúng ta có thể lấy 30 giá trị liên tiếp là features và giá trị tiếp theo làm label. Sau đó sẽ dụng mạng NN để match 30 features đến single label.
![0](images/0.png)

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)

# xem xét dữ liệu theo batch, ở đây lấy batch cho 2 dữ liệu 
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())

```
Lý do cần shuffle data do có `sequence bias`. Sequence bias is when the order of things can impact the selection of things (độ lệch (thiên vị chuỗi) - thứ tự của vật có thể ảnh hướng đến lựa chọn vật). For example, if I were to ask you your favorite TV show, and listed "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, you're probably more likely to select 'Game of Thrones' as you are familiar with it, and it's the first thing you see. Even if it is equal to the other TV shows. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up. 
