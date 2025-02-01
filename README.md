<h1 align="center">e-Commerce</h1>

<p align="center"><em>A versatile web application designed for e-commerce</em></p>

<p align="center">
    <a href="https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/Smart-Attica/e-commerce">
        <img src="https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode" alt="Open in Codespaces" />
    </a>
</p>

## :rocket: Running the project

1. **Clone the repository** and navigate to the project directory.

   ```shell
   git clone https://github.com/Smart-Attica/e-commerce.git
   cd e-commerce
   ```

2. **Edit the `.env` file**.

    ```shell
    BACKEND_APPLICATION__ENV="development"

    POSTGRES_DB='e-commerce'
    POSTGRES_USER='guest'
    POSTGRES_PASSWORD=')M8z*yss$cRxw7(&'

    BACKEND_DATABASE__URI = 'postgresql+asyncpg://guest:)M8z*yss$cRxw7(&@database:5432/e-commerce'
    ```

 3. **Build and start the services** using Docker Compose.

   ```shell
   docker-compose up -d
   ```

## :book: Exploring the Documentation

The project's documentation is automatically generated from Python docstrings using [`MkDocs`](https://www.mkdocs.org/) and [`mkdocstrings`](https://mkdocstrings.github.io/). To view it, simply run `poe docs` in your terminal which amounts to running `mkdocs serve` and opening [`http://localhost:8000`](http://localhost:8000) in your browser.
