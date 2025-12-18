# ROS 2 Nervous System Book

This educational resource introduces ROS 2 as the core nervous system of humanoid robots, enabling communication, control, and embodiment by connecting AI agents to physical actuators and sensors.

## About This Book

This book is designed for AI and software engineering students transitioning into Physical AI and humanoid robotics. It covers:

- ROS 2 fundamentals and architecture
- Connecting AI agents to robot controllers using Python
- Modeling humanoid robot bodies with URDF
- Integration with simulation environments like Gazebo and Isaac

### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

```
$ GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

### Contributing

We welcome contributions to improve the content and examples. Please follow the standard fork-and-pull request workflow.
