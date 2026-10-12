## Tổng quan

Tin sai lệch ít khi bịa ra toàn bộ một câu chuyện. Thường thì nó giữ gần hết nội dung của một bài báo thật, và chỉ đổi một chi tiết, ví dụ một con số, một cái tên, hoặc quan hệ giữa hai sự việc. Vì vậy, muốn biết một thông tin có đúng hay không, người ta đối chiếu thông tin đó với bài báo gốc.

Bài này yêu cầu các bạn xây dựng một hệ thống tự động làm việc đối chiếu đó. Hệ thống nhận vào một **nhận định**, tức một câu khẳng định về một sự việc, cùng với **bài báo** mà nhận định nói tới. Hệ thống phải cho biết bài báo **xác nhận** nhận định, **bác bỏ** nhận định, hay **không có đủ thông tin** để kết luận.

Bài được chấm bằng **macro-F1**. Với mỗi nhãn, điểm F1 cho biết hệ thống nhận ra các nhận định có nhãn đó tốt đến mức nào. Macro-F1 là trung bình cộng điểm F1 của ba nhãn, nên hệ thống phải dự đoán tốt cả ba nhãn thì mới được điểm cao.

## Nhiệm vụ

Với mỗi nhận định của tập test, hệ thống nhận đầu vào và trả về đầu ra như sau:

- **Đầu vào** là câu nhận định (cột `claim`) và mã của bài báo mà nhận định nói tới (cột `article_id`). Nội dung đầy đủ của bài báo nằm trong `articles.csv` (giai đoạn public) hoặc `private_articles.csv` (giai đoạn private).
- **Đầu ra** là một trong ba nhãn ở bảng dưới đây.

| Nhãn | Ý nghĩa |
|---|---|
| `SUPPORTED` | Bài báo có thông tin cho thấy nhận định là đúng. |
| `REFUTED` | Bài báo có thông tin mâu thuẫn với nhận định. Ví dụ, bài báo ghi một con số, một cái tên hoặc một quan hệ giữa các sự việc khác với nhận định. |
| `NEI` | Bài báo không có đủ thông tin để kết luận nhận định đúng hay sai. NEI là viết tắt của cụm từ tiếng Anh *not enough information*. |

### Ví dụ

Ba nhận định dưới đây nằm trong tập train, và cùng được viết từ một bài báo của báo Người lao động. Trong tập train, mỗi nhận định còn đi kèm một **câu bằng chứng**, tức câu trong bài báo mà người gán nhãn đã dựa vào để chọn nhãn.

**Ví dụ 1, nhãn `SUPPORTED`**

- Nhận định: "Sau tiếng động mạnh người dân phát hiện một người đàn ông tử vong bên cạnh xe máy."
- Câu bằng chứng: "Nghe tiếng động rất mạnh, người dân chạy tới thì thấy một người đàn ông khoảng 50 tuổi đã tử vong bên cạnh chiếc xe máy"

Nhận định nói lại đúng nội dung của câu bằng chứng, nên có nhãn `SUPPORTED`.

**Ví dụ 2, nhãn `REFUTED`**

- Nhận định: "Tại hiện trường ghi nhận một người phụ nữ cùng với một chiếc xe máy bị hư hỏng nặng."
- Câu bằng chứng: "Ghi nhận tại hiện trường cho thấy người đàn ông cùng chiếc xe máy hư hỏng nặng, nằm phía sau xe ben."

Nhận định viết "một người phụ nữ", trong khi bài báo viết "người đàn ông". Bài báo mâu thuẫn với nhận định, nên nhận định có nhãn `REFUTED`.

**Ví dụ 3, nhãn `NEI`**

- Nhận định: "Sau 30 phút thì lực lượng chức năng đã có mặt để giải quyết."
- Câu bằng chứng: "Công an huyện Củ Chi sau đó có mặt khám nghiệm hiện trường, trích xuất camera an ninh làm rõ vụ việc"

Bài báo có nói công an tới hiện trường, nhưng không nói họ tới sau bao lâu. Vì vậy, không thể biết chi tiết "sau 30 phút" đúng hay sai, và nhận định có nhãn `NEI`.

## Tóm tắt dữ liệu

- Tập train có 4.434 nhận định đã có nhãn, được viết từ 625 bài báo.
- **Public test** có 817 nhận định, được viết từ 208 bài báo. Các bạn có `public_test.csv` và bài báo tương ứng trong `articles.csv` từ đầu giờ thi.
- **Private test** có 808 nhận định, được viết từ 208 bài báo khác. Dữ liệu nằm trong `private_test.zip` (có mật khẩu). Ban tổ chức công bố mật khẩu khoảng **một giờ cuối**; giải nén vào cùng thư mục với notebook để được `private_test.csv` và `private_articles.csv`.
- Tập train, public test và private test dùng **các bài báo khác nhau**. Khi dự đoán, hệ thống luôn phải làm việc với những bài báo mà nó chưa gặp lúc huấn luyện.
- Các bài báo lấy từ 9 báo điện tử Việt Nam và thuộc nhiều chủ đề, như thời sự, kinh tế, pháp luật, giáo dục, thể thao và văn hoá. Mỗi bài dài trung bình khoảng 3.100 ký tự.

### Những điều cần biết về dữ liệu

- Câu bằng chứng (cột `evidence`) **chỉ có trong tập train**. Ở tập test, hệ thống phải tự tìm những câu liên quan trong bài báo.
- Câu bằng chứng do người gán nhãn chép ra, nên đôi khi không giống hệt câu nào trong bài báo.
- Nhận định có nhãn `REFUTED` và nhận định có nhãn `NEI` cũng dùng lại rất nhiều từ của bài báo. Vì vậy, nếu chỉ đếm số từ chung giữa nhận định và bài báo thì chưa phân biệt được ba nhãn.
- Số nhận định của ba nhãn gần bằng nhau, cả trong tập train lẫn tập test.

## Các tệp dữ liệu

| Tệp | Nội dung |
|---|---|
| `train.csv` | 4.434 nhận định đã có nhãn, dùng để huấn luyện. |
| `articles.csv` | 833 bài báo: 625 bài của train và 208 bài của public test. |
| `public_test.csv` | 817 nhận định public test cần dự đoán nhãn. |
| `sample_submission.csv` | Một tệp nộp mẫu đúng định dạng cho public test. |
| `private_test.zip` | Tập private test, có mật khẩu. Khi giải nén được `private_test.csv` (808 nhận định) và `private_articles.csv` (208 bài báo). |
| `notebook_mau.ipynb` | Notebook mẫu, gồm code baseline, phần tự chấm trên tập validation và phần ghi tệp nộp. |
| `scorer.py` | Trình chấm để các bạn tự chấm điểm trên máy của mình. |

### Tệp `train.csv`

Mỗi dòng của tệp là một nhận định đã có nhãn. Tệp có năm cột:

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã của nhận định, ví dụ `f00001`. Mỗi nhận định có một mã riêng. |
| `article_id` | Mã của bài báo mà nhận định nói tới. Nội dung bài báo nằm trong `articles.csv`. |
| `claim` | Câu nhận định cần kiểm chứng. |
| `evidence` | Câu bằng chứng, tức câu trong bài báo mà người gán nhãn đã dựa vào để chọn nhãn. Cột này chỉ có trong tập train. |
| `label` | Nhãn của nhận định, là `SUPPORTED`, `REFUTED` hoặc `NEI`. |

### Tệp `articles.csv`

Mỗi dòng của tệp là một bài báo. Tệp chứa 833 bài báo, gồm 625 bài báo của tập train và 208 bài báo của public test. Bài báo của private test **không** nằm trong tệp này; chúng nằm trong `private_articles.csv` sau khi giải nén `private_test.zip`. Tệp có bốn cột:

| Cột | Ý nghĩa |
|---|---|
| `article_id` | Mã của bài báo. |
| `source` | Tên báo điện tử đã đăng bài, ví dụ VnExpress hoặc Tuổi Trẻ. |
| `topic` | Chủ đề của bài, viết bằng chữ thường, ví dụ `giáo dục` hoặc `thể thao`. |
| `text` | Toàn văn bài báo. |

### Tệp `public_test.csv`

Tệp có 817 nhận định public test. Tệp có ba cột `id`, `article_id` và `claim`, với cùng ý nghĩa như trong `train.csv`. Tập test không có cột `evidence` và cột `label`.

### Tệp `private_test.zip`

Tệp zip có mật khẩu. Ban tổ chức công bố mật khẩu trong giai đoạn private. Giải nén vào cùng thư mục với notebook:

```
unzip private_test.zip
```

Sau khi giải nén có hai tệp: `private_test.csv` (808 nhận định, cùng ba cột với `public_test.csv`) và `private_articles.csv` (208 bài báo, cùng bốn cột với `articles.csv`).

### Tệp `sample_submission.csv`

Đây là một tệp nộp mẫu đúng định dạng cho public test, trong đó mọi nhận định đều được gán nhãn `SUPPORTED`. Tệp này hợp lệ, nhưng chỉ được khoảng 0,17 điểm.

## Baseline

Baseline là một cách giải đơn giản do ban tổ chức viết sẵn, để các bạn có ngay một tệp nộp đúng định dạng và một mức điểm để so sánh. Code của baseline nằm trong notebook mẫu `notebook_mau.ipynb`.

Baseline của bài này làm ba bước:

1. Chia bài báo thành từng câu, rồi dùng TF-IDF để tìm câu giống nhận định nhất. TF-IDF biểu diễn mỗi câu bằng trọng số của các từ trong câu, trong đó từ càng hiếm thì trọng số càng cao.
2. Tính mười đặc trưng dạng số cho mỗi nhận định. Ví dụ, có đặc trưng đo độ giống nhau giữa nhận định và câu vừa tìm được, có đặc trưng đo tỉ lệ từ của nhận định có mặt trong bài báo, và có đặc trưng đếm số con số có trong nhận định nhưng không có trong bài báo.
3. Huấn luyện model hồi quy logistic, một model phân loại đơn giản, trên mười đặc trưng đó để dự đoán nhãn.

Notebook mẫu tách khoảng 20 % số bài báo của tập train làm **tập validation**, tức phần dữ liệu có nhãn được để riêng ra, không dùng để huấn luyện mà dùng để tự chấm điểm. Validation được tách **theo bài báo**, nghĩa là mọi nhận định của cùng một bài báo đều nằm ở phần train hoặc đều nằm ở validation. Cả notebook chạy trên CPU trong khoảng nửa phút. Ban tổ chức đo được macro-F1 khoảng **0,57** trên tập validation và khoảng **0,60** trên tập test.

### Trình chấm `scorer.py`

Trình chấm này tính điểm bằng đúng công thức macro-F1 của bài. Các bạn nên dùng nó để tự chấm dự đoán trên tập validation trước khi nộp. Cách dùng như sau:

```
python3 scorer.py du_doan_validation.csv --gt validation.csv
python3 scorer.py du_doan_validation.csv --gt validation.csv --show 10
```

Trong hai lệnh trên, `du_doan_validation.csv` là tệp dự đoán, có hai cột `id` và `label` giống tệp CSV trong bài nộp. Tệp `validation.csv` là tệp đáp án, cũng có hai cột `id` và `label`. Ngoài macro-F1, trình chấm còn in precision, recall và F1 của từng nhãn, cùng ma trận nhầm lẫn. Lệnh thứ hai có thêm tuỳ chọn `--show 10` để in 10 nhận định bị đoán sai. Tuỳ chọn này cần tệp đáp án có thêm cột `claim`.

## Định dạng bài nộp

Tệp nộp là một tệp **zip**, mã hoá UTF-8, chứa **đúng một tệp CSV**. Tệp CSV có dòng tiêu đề và có đúng số dòng của tập test **đang chấm**: 817 dòng cho public test, hoặc 808 dòng cho private test, mỗi dòng một `id`. Tệp CSV có hai cột là `id` và `label`. Ví dụ:

```
id,label
f00002,SUPPORTED
f00003,SUPPORTED
f00006,SUPPORTED
```

Notebook mẫu ghi tệp `bai1_submission.csv` rồi nén thành `submission.zip`. Đó là tệp để nộp.

Trình chấm xử lý các lỗi thường gặp như sau:

- Nếu tệp thiếu dòng của một `id`, nhận định đó được tính là dự đoán sai.
- Cột `label` chỉ nhận ba giá trị `SUPPORTED`, `REFUTED` và `NEI`, không phân biệt chữ hoa hay chữ thường. Mọi giá trị khác được tính là dự đoán sai.
- Nếu tệp không có cột `id` hoặc cột `label`, bài nộp được 0 điểm.
- Nếu tệp zip không chứa đúng một tệp CSV, bài nộp không hợp lệ.

## Phương pháp đánh giá

### Macro-F1

Trình chấm so nhãn dự đoán với nhãn thật của từng nhận định trong tập test. Với mỗi nhãn, ví dụ nhãn `NEI`, trình chấm tính ba số sau:

- **Precision** của nhãn `NEI` là tỉ lệ dự đoán đúng trong số các nhận định được dự đoán là `NEI`. Precision thấp nghĩa là hệ thống hay gán nhãn `NEI` cho những nhận định có nhãn khác.
- **Recall** của nhãn `NEI` là tỉ lệ nhận định được dự đoán đúng trong số các nhận định có nhãn thật là `NEI`. Recall thấp nghĩa là hệ thống bỏ sót nhiều nhận định `NEI`.
- **F1** của nhãn `NEI` kết hợp precision và recall theo công thức dưới đây. F1 chỉ cao khi cả precision lẫn recall đều cao.

```
F1 = 2 × precision × recall / (precision + recall)
```

Macro-F1 là trung bình cộng F1 của ba nhãn:

```
Điểm = (F1 của SUPPORTED + F1 của REFUTED + F1 của NEI) / 3
```

Điểm nằm trong khoảng từ 0 đến 1. Ba nhãn có trọng số bằng nhau. Vì vậy, một hệ thống không bao giờ dự đoán `NEI` sẽ mất trọn một phần ba số điểm, dù hai nhãn còn lại được dự đoán tốt.

**Ví dụ.** Giả sử tập test chỉ có sáu nhận định, với nhãn thật và nhãn dự đoán như bảng sau.

| Nhận định | Số 1 | Số 2 | Số 3 | Số 4 | Số 5 | Số 6 |
|---|---|---|---|---|---|---|
| Nhãn thật | SUPPORTED | SUPPORTED | REFUTED | REFUTED | NEI | NEI |
| Nhãn dự đoán | SUPPORTED | SUPPORTED | SUPPORTED | REFUTED | REFUTED | NEI |

- Nhãn `SUPPORTED` được dự đoán 3 lần và đúng 2 lần, nên precision là 2/3. Cả 2 nhận định có nhãn thật `SUPPORTED` đều được đoán đúng, nên recall là 2/2. F1 của nhãn này là 0,80.
- Nhãn `REFUTED` được dự đoán 2 lần và đúng 1 lần, nên precision là 1/2. Trong 2 nhận định có nhãn thật `REFUTED`, chỉ 1 nhận định được đoán đúng, nên recall là 1/2. F1 của nhãn này là 0,50.
- Nhãn `NEI` được dự đoán 1 lần và lần đó đúng, nên precision là 1/1. Trong 2 nhận định có nhãn thật `NEI`, chỉ 1 nhận định được đoán đúng, nên recall là 1/2. F1 của nhãn này là 0,67.
- Macro-F1 là (0,80 + 0,50 + 0,67) / 3, tức khoảng 0,66.

## Lưu ý

- Các bạn khi làm cần tuân thủ về mô hình pretrained mà BTC cuộc thi có nói tới [Thể lệ thi Olympic AI Sinh viên](https://drive.google.com/file/d/1FR4V4i6eWMOS8lJshQwjgUKC41OvBcT0/view?usp=sharing)
- Cụ thể, các bạn được dùng các model pretrained cơ bản do các thư viện trong danh sách của thể lệ cung cấp, ví dụ vector từ của fastText hoặc gensim. Các bạn không được dùng model ngôn ngữ pretrained dạng Transformer, như PhoBERT, XLM-R, multilingual-e5 hay ViT5, và không được dùng LLM.
- Chỉ import các thư viện trong danh sách của thể lệ, và không cài thêm thư viện.
- Được dùng dữ liệu của cả ba bài trong contest. Không được dùng dữ liệu ngoài contest, kể cả bộ dữ liệu ViFactCheck gốc và các bài báo gốc trên trang báo.
- Không được dùng dữ liệu của public test hoặc private test để huấn luyện, dưới bất kỳ hình thức nào. Ví dụ, không được học TF-IDF trên văn bản của tập test. Dùng TF-IDF đã học từ tập train để biểu diễn câu của tập test thì được phép, như baseline đang làm.

## Nguồn dữ liệu

Dữ liệu lấy từ bộ ViFactCheck (Hoa và cộng sự, AAAI 2025), dùng giấy phép MIT. Các nhận định của bộ dữ liệu được viết từ bài báo của 9 báo điện tử Việt Nam. Ban tổ chức đã chuẩn hoá văn bản và chia lại dữ liệu theo bài báo. Ở tập test, mỗi bài báo chỉ giữ lại từ 3 đến 5 nhận định, được chọn ngẫu nhiên.
