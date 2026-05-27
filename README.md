Dự án cuối kỳ môn Học máy (PHY3638) đề tài "Ứng dụng học sâu cho bài toán hồi quy dự đoán tuổi của người"
## 📌 Giới thiệu
Dự án tập trung nghiên cứu và xây dựng mô hình dự đoán tuổi của con người từ ảnh khuôn mặt bằng phương pháp học sâu (Deep Learning).  
Bài toán được tiếp cận dưới dạng **hồi quy (Regression)** thay vì phân loại, trong đó tuổi được xem là một biến liên tục.

Mô hình đề xuất kết hợp:
- **CNN (Convolutional Neural Network)** để trích xuất đặc trưng ảnh khuôn mặt.
- **Ridge Regression** để thực hiện dự đoán tuổi và giảm hiện tượng overfitting.

---
### Mục tiêu

- Đọc và tiền xử lý dữ liệu ảnh khuôn mặt
- Trích xuất đặc trưng bằng CNN
- Dự đoán tuổi bằng Ridge Regression
- Đánh giá mô hình thông qua:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- Trực quan hóa:
  - Phân bố dữ liệu tuổi
  - Loss curve
  - Biểu đồ dự đoán và giá trị thực tế

---
### Thư viện sử dụng

| Thư viện | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| PyTorch | Xây dựng và huấn luyện CNN |
| Scikit-learn | Ridge Regression và đánh giá mô hình |
| OpenCV | Xử lý ảnh |
| NumPy & Pandas | Xử lý dữ liệu |
| Matplotlib & Seaborn | Vẽ đồ thị trực quan |
| Kaggle Dataset | Bộ dữ liệu khuôn mặt |

---
### Những khó khăn gặp phải

Một số khó khăn trong quá trình thực hiện:

- Dataset bị mất cân bằng giữa các nhóm tuổi
- Chất lượng ảnh không đồng đều
- Tối ưu tham số regularization (`alpha`) cho Ridge Regression

---
### Hướng phát triển

- Dự đoán thêm giới tính và chủng tộc
- Triển khai thành ứng dụng web
- Cải thiện hiệu suất trên các nhóm tuổi ít dữ liệu

---
## 3. Cài đặt và chạy dự án
### Clone Repository

```bash
git clone https://github.com/kamenriderkuuga7805-arch/human_age_prediction_HUS.git
cd human_age_prediction_HUS

cd your-repository-name
om/kamenriderkuuga7805-arch/human_age_pr
