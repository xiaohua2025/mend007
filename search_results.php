<?php
// 数据库配置
$host = 'localhost';
$dbname = 'your_database';
$username = 'username';
$password = 'password';

try {
    // 创建数据库连接
    $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // 获取查询参数
    $name = $_GET['name'] ?? '';
    $manufacturer = $_GET['manufacturer'] ?? '';
    $location = $_GET['location'] ?? '';
    $batch_number = $_GET['batch_number'] ?? '';
    $expiration_date_start = $_GET['expiration_date_start'] ?? '';
    $expiration_date_end = $_GET['expiration_date_end'] ?? '';

    // 构建 SQL 查询
    $sql = "SELECT * FROM chemicals WHERE 1=1";
    if (!empty($name)) {
        $sql .= " AND name LIKE :name";
    }
    if (!empty($manufacturer)) {
        $sql .= " AND manufacturer LIKE :manufacturer";
    }
    if (!empty($location)) {
        $sql .= " AND location LIKE :location";
    }
    if (!empty($batch_number)) {
        $sql .= " AND batch_number LIKE :batch_number";
    }
    if (!empty($expiration_date_start)) {
        $sql .= " AND expiration_date >= :expiration_date_start";
    }
    if (!empty($expiration_date_end)) {
        $sql .= " AND expiration_date <= :expiration_date_end";
    }

    $stmt = $pdo->prepare($sql);

    // 绑定参数
    if (!empty($name)) {
        $stmt->bindValue(':name', "%$name%");
    }
    if (!empty($manufacturer)) {
        $stmt->bindValue(':manufacturer', "%$manufacturer%");
    }
    if (!empty($location)) {
        $stmt->bindValue(':location', "%$location%");
    }
    if (!empty($batch_number)) {
        $stmt->bindValue(':batch_number', "%$batch_number%");
    }
    if (!empty($expiration_date_start)) {
        $stmt->bindValue(':expiration_date_start', $expiration_date_start);
    }
    if (!empty($expiration_date_end)) {
        $stmt->bindValue(':expiration_date_end', $expiration_date_end);
    }

    // 执行查询
    $stmt->execute();
    $results = $stmt->fetchAll(PDO::FETCH_ASSOC);

    // 显示结果
    if (count($results) > 0) {
        echo "<h2>查询结果</h2>";
        echo "<table border='1'>
                <tr>
                    <th>化学品名称</th>
                    <th>类别</th>
                    <th>图片</th>
                    <th>库位</th>
                    <th>操作</th>
                    <th>生产厂家</th>
                    <th>生产批次</th>
                    <th>生产日期</th>
                    <th>失效日期</th>
                </tr>";
        foreach ($results as $row) {
            echo "<tr>
                    <td>{$row['name']}</td>
                    <td>{$row['category']}</td>
                    <td><img src='{$row['image']}' width='100'></td>
                    <td>{$row['location']}</td>
                    <td>{$row['action']}</td>
                    <td>{$row['manufacturer']}</td>
                    <td>{$row['batch_number']}</td>
                    <td>{$row['production_date']}</td>
                    <td>{$row['expiration_date']}</td>
                  </tr>";
        }
        echo "</table>";
    } else {
        echo "没有找到匹配的记录。";
    }
} catch (PDOException $e) {
    echo "错误: " . $e->getMessage();
}
?>