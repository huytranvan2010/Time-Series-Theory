# Time-Series-Theory

# Moving average and diferencing
Một cách dự đoán đơn giản là tính `moving average`. Ví dụ gia trị tiếp theo bằng trung bình của 30 ngày trước đó. Cách này loại bỏ rất nhiều nhiễu, nó không dự đoán được xu hướng và tính thời vụ. 

Một cách để tránh được điều này là loại bỏ `trend` và `seasonality` khỏi time series với kỹ thuật `differecing`. Chúng ta sẽ nghiên cứu sự khác nhau giữa giá trị tại thời điểm `t` và giá trị tại thời điểm trước đó (tùy thuộc trường hợp là ngày, tháng, năm...), khi đó chúng ta không có trend và cả seasonality. Sau đó sử dụng `moving average` để dự đoán giá trị cho time series này. Nên nhớ đây là dự đoán cho difference, muốn lấy giá trị thực tế cần cộng với giá trị tại thời điểm trước đó. Cuối cùng ta nhận được đường dự đoán có nhiều noise (noise này xuất hiện từ giá trị tại thời điểm trước đó từ time series ban đầu). Chúng ta có thể loại bỏ noise này bằng cách áp dụng `moving average` cho giá trị tại thời điểm trước.