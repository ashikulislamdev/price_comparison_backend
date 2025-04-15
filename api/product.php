<?php
include 'db.php';

header('Content-Type: application/json');

//Upload Product Image and Save Product Info
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['image'])) {
    $user_id = $_POST['user_id'];
    $target_dir = "../uploads/";
    $file_name = basename($_FILES["image"]["name"]);
    $target_file = $target_dir . time() . "_" . $file_name;

    if (move_uploaded_file($_FILES["image"]["tmp_name"], $target_file)) {
        // Dummy AI output (to be replaced with real model later)
        $product_name = "Red T-Shirt";
        $category = "Clothing";
        $color = "Red";
        $brand = "Nike";

        // Insert into products table
        $sql = "INSERT INTO products (user_id, image_url, product_name, category, color, brand) VALUES (?, ?, ?, ?, ?, ?)";
        $stmt = $conn->prepare($sql);
        $stmt->execute([$user_id, $target_file, $product_name, $category, $color, $brand]);

        $product_id = $conn->lastInsertId();

        // Dummy price comparison data (will be replaced by real scraping later)
        $stores = [
            ["store_name" => "Amazon", "price" => 29.99, "product_link" => "https://amazon.com/product123"],
            ["store_name" => "eBay", "price" => 27.50, "product_link" => "https://ebay.com/product123"]
        ];

        foreach ($stores as $store) {
            $sql = "INSERT INTO price_comparison (product_id, store_name, price, product_link) VALUES (?, ?, ?, ?)";
            $stmt = $conn->prepare($sql);
            $stmt->execute([$product_id, $store['store_name'], $store['price'], $store['product_link']]);
        }

        echo json_encode([
            "message" => "Image uploaded and product created successfully!",
            "product_id" => $product_id,
            "image_url" => $target_file
        ]);
    } else {
        echo json_encode(["error" => "File upload failed"]);
    }
    exit;
}

//Get All Products of a User
if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['user_id'])) {
    $user_id = $_GET['user_id'];

    $sql = "SELECT * FROM products WHERE user_id=?";
    $stmt = $conn->prepare($sql);
    $stmt->execute([$user_id]);
    $products = $stmt->fetchAll(PDO::FETCH_ASSOC);

    echo json_encode($products);
    exit;
}

//Get Price Comparison Data for a Product
if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['product_id'])) {
    $product_id = $_GET['product_id'];

    $sql = "SELECT * FROM price_comparison WHERE product_id=?";
    $stmt = $conn->prepare($sql);
    $stmt->execute([$product_id]);
    $prices = $stmt->fetchAll(PDO::FETCH_ASSOC);

    echo json_encode($prices);
    exit;
}

echo json_encode(["error" => "Invalid request"]);
?>
