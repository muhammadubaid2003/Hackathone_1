/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

module.exports = {
  // Combined sidebar for all modules
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI Fundamentals',
      items: [
        'modules/ros2-nervous-system/ros2-nervous-system-overview',
        'modules/ros2-nervous-system/introduction',
        'modules/ros2-nervous-system/python-agents',
        'modules/ros2-nervous-system/urdf-modeling',
        'modules/ros2-nervous-system/glossary',
        'modules/ros2-nervous-system/summary'
      ],
    },
    {
      type: 'category',
      label: 'Digital Twin Technologies',
      items: [
        'modules/digital-twin/introduction',
        'modules/digital-twin/physics-simulation-gazebo',
        'modules/digital-twin/high-fidelity-unity',
        'modules/digital-twin/sensor-simulation',
        'modules/digital-twin/glossary',
        'modules/digital-twin/summary'
      ],
    },
    {
      type: 'category',
      label: 'AI & Robotics Intelligence',
      items: [
        'modules/ai-robot-brain/introduction',
        'modules/ai-robot-brain/isaac-sim-synthetic-data',
        'modules/ai-robot-brain/isaac-ros-vslam',
        'modules/ai-robot-brain/nav2-humanoid-navigation',
        'modules/ai-robot-brain/glossary',
        'modules/ai-robot-brain/summary'
      ],
    },
    {
      type: 'category',
      label: 'Multimodal AI Systems',
      items: [
        'modules/vla-integration/introduction',
        'modules/vla-integration/voice-to-action-pipelines',
        'modules/vla-integration/cognitive-planning-llms',
        'modules/vla-integration/capstone-autonomous-humanoid',
        'modules/vla-integration/glossary',
        'modules/vla-integration/summary'
      ],
    },
  ],

  // Additional sidebars can be added here
};
