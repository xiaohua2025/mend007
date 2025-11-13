<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>添加化学品</title>
</head>
<body>
    <h1>添加化学品</h1>
    <form action="save_chemical.php" method="post" enctype="multipart/form-data">
        <label for="name">化学品名称:</label>
        <input type="text" id="name" name="name" required><br><br>

        <label for="category">化学品类别:</label>
        <input type="text" id="category" name="category" required><br><br>

        <label for="image">化学品图片:</label>
        <input type="file" id="image" name="image"><br><br>

        <label for="location">库位:</label>
        <input type="text" id="location" name="location" required><br><br>

        <label for="action">操作:</label>
        <select id="action" name="action" required>
            <option value="入库">入库</option>
            <option value="出库">出库</option>
        </select><br><br>

        <label for="manufacturer">生产厂家:</label>
        <input type="text" id="manufacturer" name="manufacturer" required><br><br>

        <label for="batch_number">生产批次:</label>
        <input type="text" id="batch_number" name="batch_number" required><br><br>

        <label for="production_date">生产日期:</label>
        <input type="date" id="production_date" name="production_date" required><br><br>

        <label for="expiration_date">失效日期:</label>
        <input type="date" id="expiration_date" name="expiration_date" required><br><br>

        <button type="submit">提交</button>
    </form>
</body>
</html>