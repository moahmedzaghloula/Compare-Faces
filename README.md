# CompareFaces API

## Overview

This project provides a FastAPI-based API for comparing faces in two uploaded images. It leverages MTCNN for efficient face detection and alignment, and FaceNet for generating robust face embeddings. The API determines if two faces belong to the same person based on the cosine distance between their embeddings.

## Features

*   **Face Detection & Alignment**: Utilizes MTCNN to accurately detect and align faces within images.
*   **Face Embedding Generation**: Employs FaceNet to create high-dimensional embeddings for detected faces.
*   **Face Comparison**: Compares face embeddings using cosine similarity to determine if two individuals are the same.
*   **FastAPI**: Provides a modern, fast (high-performance) web framework for building APIs with Python 3.7+.

## Technologies Used

*   **FastAPI**: Web framework for building the API.
*   **Uvicorn**: ASGI server for running the FastAPI application.
*   **OpenCV (cv2)**: For image processing tasks.
*   **MTCNN**: Multi-task Cascaded Convolutional Networks for face detection and alignment.
*   **Keras-Facenet**: Implementation of FaceNet for generating face embeddings.
*   **SciPy**: For scientific computing, specifically for cosine distance calculation.
*   **NumPy**: For numerical operations, especially array manipulation.
*   **Pillow (PIL)**: For image handling.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/moahmedzaghloula/Compare-Faces.git
    cd CompareFaces
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the API

To start the FastAPI application, run the following command:

```bash
python compare_fast_api.py
```

The API will be accessible at `http://0.0.0.0:5000`.

### API Endpoint

*   **POST `/compare_faces`**
    *   **Description**: Compares two uploaded images to determine if they contain the same person.
    *   **Request Body**: Expects two image files as `UploadFile` (multipart/form-data).
        *   `image1`: The first image file.
        *   `image2`: The second image file.
    *   **Responses**:
        *   `200 OK`: If faces are detected and compared successfully.
            ```json
            {
                "result": "They are the same person."
            }
            ```
            or
            ```json
            {
                "result": "They are not the same person."
            }
            ```
        *   `400 Bad Request`: If a face is not detected in one or both images.
            ```json
            {
                "detail": "Face not detected in one or both images"
            }
            ```

### Example Usage (using `curl`)

```bash
curl -X POST "http://localhost:5000/compare_faces" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image1=@/home/mohamed-zaghloula/Pictures/Screenshots/Screenshot From 2025-07-18 16-49-02.png" \
  -F "image2=@/home/mohamed-zaghloula/Pictures/Screenshots/Screenshot From 2025-07-18 16-49-13.png"
```



## Project Structure

```
CompareFaces/
├── compare_fast_api.py
├── requirements.txt
└── render.yaml
```

*   `compare_fast_api.py`: Contains the FastAPI application logic, including face detection, embedding, and comparison functions.
*   `requirements.txt`: Lists all Python dependencies required to run the project.
*   `render.yaml`: (Optional) Configuration file for deployment on Render.com.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable).


## Example Usage with Images

To illustrate the face comparison functionality, consider the following example images:

### Image 1: Messi (Younger)

![Messi Younger](/home/mohamed-zaghloula/Pictures/Screenshots/Screenshot From 2025-07-18 16-49-02.png)

### Image 2: Messi (Older)

![Messi Older](/home/mohamed-zaghloula/Pictures/Screenshots/Screenshot From 2025-07-18 16-49-13.png)

When these two images are passed to the `/compare_faces` endpoint, the API will analyze them and return a result indicating whether they are the same person. Given that both images are of Lionel Messi, the expected output would be:

```json
{
    "result": "They are the same person."
}
```

This demonstrates the API's ability to correctly identify the same individual across different images, even with variations in age or appearance.