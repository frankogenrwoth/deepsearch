# Semantic Search API

This project provides an API to enable efficient semantic search for large datasets. The solution allows clients to upload their data, which is indexed and stored in a non-relational database (Firebase). Clients can then integrate an API on their websites to implement powerful semantic search functionality.

## Features
- Upload client data for indexing.
- Perform semantic search on uploaded data.
- API-first design for easy integration into client websites.

## Technologies Used
- **Django**: Backend framework for rapid development.
- **Django REST Framework**: API development.
- **Firebase**: Data storage and indexing.
- **Sentence Transformers**: For generating semantic embeddings.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/semantic-search-api.git
   cd semantic-search-api
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Firebase service account key:
   - Place the Firebase credentials JSON file in the project directory.
   - Update the `settings.py` file to include Firebase initialization.

5. Run migrations:
   ```bash
   python manage.py migrate
   ```

6. Start the development server:
   ```bash
   python manage.py runserver
   ```

## API Endpoints
### 1. **Upload Data**
- **URL**: `/api/upload/`
- **Method**: `POST`
- **Description**: Upload client data for indexing.
- **Request Body**:
  ```json
  {
    "title": "Document Title",
    "content": "Document content goes here."
  }
  ```

### 2. **Search**
- **URL**: `/api/search/`
- **Method**: `POST`
- **Description**: Perform semantic search.
- **Request Body**:
  ```json
  {
    "query": "Search query here."
  }
  ```

## Future Enhancements
- Add query analytics for clients.
- Support hybrid search (semantic + keyword).
- Add a client dashboard for managing data.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## License
This project is licensed under the MIT License.