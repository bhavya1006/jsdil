"""Starts the REST service, after checking that everything it needs is on disk."""
import os
import sys

from config import config


def main():
    """Start the service, or name what is missing and what produces it."""
    missing = [(path, producer) for path, producer in config.SERVICE_REQUIRED_FILES
               if not os.path.exists(path)]
    if missing:
        print("The service cannot start.")
        for path, producer in missing:
            print(f"  {path} is missing; it comes from {producer}")
        return 1
    import uvicorn

    from api.routes import service_application

    print(f"Starting {config.API_TITLE} on port {config.API_PORT}. Interactive documentation: "
          f"http://localhost:{config.API_PORT}/docs")
    uvicorn.run(service_application, host=config.API_HOST, port=config.API_PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
