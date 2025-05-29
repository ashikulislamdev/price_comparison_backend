<?php
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $data = json_decode(file_get_contents("php://input"));
    if (!isset($data->product_name)) {
        echo json_encode(["error" => "Missing product_name"]);
        exit;
    }

    $product = urlencode($data->product_name);

    // Dummy response for testing 
    $results = [
        [
            "name" => "$data->product_name - Amazon",
            "price" => "$49.99",
            "link" => "https://www.amazon.com/s?k=$product"
        ],
        [
            "name" => "$data->product_name - Daraz",
            "price" => "$45.50",
            "link" => "https://www.daraz.com.bd/catalog/?q=$product"
        ]
    ];

    echo json_encode([
        "product" => $data->product_name,
        "results" => $results
    ]);
    exit;
}

echo json_encode(["error" => "Invalid request method"]);
?>