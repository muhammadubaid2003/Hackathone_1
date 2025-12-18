# Quickstart: ROS 2 Nervous System Module

## Prerequisites

- Node.js 18+ installed
- Basic understanding of JavaScript/Markdown
- Git installed for version control

## Setup Instructions

1. **Install Docusaurus globally (if not already installed):**
   ```bash
   npm install -g @docusaurus/core@latest
   ```

2. **Navigate to your project directory:**
   ```bash
   cd your-project-directory
   ```

3. **Install project dependencies:**
   ```bash
   npm install
   ```

4. **Start the development server:**
   ```bash
   npm start
   ```

5. **Open your browser to:**
   ```
   http://localhost:3000
   ```

## Adding the ROS 2 Module Content

1. **Create the module directory:**
   ```bash
   mkdir -p docs/modules/ros2-nervous-system
   ```

2. **Add the three chapter files:**
   - `docs/modules/ros2-nervous-system/introduction.md`
   - `docs/modules/ros2-nervous-system/python-agents.md`
   - `docs/modules/ros2-nervous-system/urdf-modeling.md`

3. **Example chapter file structure:**
   ```markdown
   ---
   id: introduction
   title: Introduction to ROS 2 as a Robotic Nervous System
   sidebar_position: 1
   ---

   # Introduction to ROS 2 as a Robotic Nervous System

   Content goes here...
   ```

## Building for Production

```bash
npm run build
```

The static site will be generated in the `build/` directory and ready for deployment to GitHub Pages.

## Deployment

1. **Build the site:**
   ```bash
   npm run build
   ```

2. **Deploy to GitHub Pages using Docusaurus command:**
   ```bash
   npm run deploy
   ```

## Customization

- Modify `docusaurus.config.js` to update site configuration
- Add custom components to the `src/` directory
- Place static assets in the `static/` directory
- Customize styles in the `src/css/` directory