<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>高级查询</title>
</head>
<body>
    <h1>高级查询</h1>
    <form action="search_results.php" method="get">
        <label for="name">化学品名称:</label>
        <input type="text" id="name" name="name"><br><br>

        <label for="manufacturer">生产厂家:</label>
        <input type="text" id="manufacturer" name="manufacturer"><br><br>

        <label for="location">库位:</label>
        <input type="text" id="location" name="location"><br><br>

        <label for="batch_number">生产批次:</label>
        <input type="text" id="batch_number" name="batch_number"><br><br>

        <label for="expiration_date_start">失效日期范围:</label>
        <input type="date" id="expiration_date_start" name="expiration_date_start">
        至
        <input type="date" id="expiration_date_end" name="expiration_date_end"><br><br>

        <button type="submit">查询</button>
    </form>
</body>
</html>