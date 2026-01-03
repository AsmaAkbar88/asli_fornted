// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // Manual sidebar for the Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category', 
      label: 'Introduction',
      items: ['intro'],
      link: {
        type: 'doc',
        id: 'intro',
      },
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/basics',
        'module-1-ros2/nodes',
        'module-1-ros2/topics',
        'module-1-ros2/services'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo/Unity',
      items: [
        'module-2-gazebo/index',
        'module-2-gazebo/simulation',
        'module-2-gazebo/models',
        'module-2-gazebo/physics'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac',
      items: [
        'module-3-nvidia-isaac/index',
        'module-3-nvidia-isaac/setup',
        'module-3-nvidia-isaac/perception',
        'module-3-nvidia-isaac/control'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA (Vision-Language-Action)',
      items: [
        'module-4-vla/index',
        'module-4-vla/vision-language',
        'module-4-vla/manipulation',
        'module-4-vla/examples'
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/index',
        'capstone/integration'
      ],
    },
    {
      type: 'category',
      label: 'Hardware Requirements',
      items: [
        'hardware/index',
        'hardware/requirements',
        'hardware/deployment'
      ],
    },
    {
      type: 'category',
      label: 'Cloud Deployment',
      items: [
        'cloud/index',
        'cloud/options'
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/glossary'],
    },
  ],
};

export default sidebars;
