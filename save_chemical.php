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

    // 处理表单数据
    $name = $_POST['name'];
    $category = $_POST['category'];
    $location = $_POST['location'];
    $action = $_POST['action'];
    $manufacturer = $_POST['manufacturer'];
    $batch_number = $_POST['batch_number'];
    $production_date = $_POST['production_date'];
    $expiration_date = $_POST['expiration_date'];

    // 处理上传的图片
    $image = '';
    if ($_FILES['image']['error'] === UPLOAD_ERR_OK) {
        $upload_dir = 'uploads/';
        $image = $upload_dir . basename($_FILES['image']['name']);
        move_uploaded_file($_FILES['image']['tmp_name'], $image);
    }

    // 插入数据到数据库
    $sql = "INSERT INTO chemicals (name, category, image, location, action, manufacturer, batch_number, production_date, expiration_date)
            VALUES (:name, :category, :image, :location, :action, :manufacturer, :batch_number, :production_date, :expiration_date)";
    $stmt = $pdo->prepare($sql);

    $stmt->bindParam(':name', $name);
    $stmt->bindParam(':category', $category);
    $stmt->bindParam(':image', $image);
    $stmt->bindParam(':location', $location);
    $stmt->bindParam(':action', $action);
    $stmt->bindParam(':manufacturer', $manufacturer);
    $stmt->bindParam(':batch_number', $batch_number);
    $stmt->bindParam(':production_date', $production_date);
    $stmt->bindParam(':expiration_date', $expiration_date);

    $stmt->execute();

    echo "化学品信息保存成功！";
} catch (PDOException $e) {
    echo "错误: " . $e->getMessage();
}
?>