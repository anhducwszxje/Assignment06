# Báo cáo Assignment 06: Recurrent Neural Networks

- `main.tex`: toàn bộ nội dung, bảng, công thức và tài liệu tham khảo.
- `images/`: logo, khung bìa, hình kết quả và 21 ảnh mã nguồn trích từ năm notebook.
- `Assignment_06_LaTeX_Report.zip`: gói nguồn để nhập vào Overleaf.

## Biên dịch

Chọn **XeLaTeX** và đặt `main.tex` làm tài liệu chính. Giữ thư mục `images` cạnh `main.tex`. Biên dịch hai lần để cập nhật mục lục và tham chiếu; chạy thêm một lần nếu trình biên dịch còn yêu cầu cập nhật nhãn.

```text
xelatex main.tex
xelatex main.tex
```

Mẫu dùng Times New Roman nếu font có sẵn, hoặc TeX Gyre Termes làm font thay thế. Không cần BibTeX vì tài liệu tham khảo nằm trong `main.tex`.

Bản PDF đã kiểm tra được lưu tại `output/pdf/assignment06_technical_report.pdf`, tính từ thư mục gốc dự án INTEL_SYS.

Báo cáo có 60 trang, 21 bảng và 41 hình. Các ảnh mã nguồn ghi tên notebook, chỉ số thực thi và số dòng của cell; phần giải thích đi kèm làm rõ dữ liệu đầu vào, thao tác xử lý và đầu ra. Chương triển khai trình bày suy luận trong notebook và thiết kế dịch vụ; hiện chưa có ứng dụng web/API độc lập.
