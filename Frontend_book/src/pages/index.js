import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/modules/ros2-nervous-system/ros2-nervous-system-overview">
              Start Reading - Chapter 1
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

// Textbook-style chapter overview component
function ChapterOverview() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--3">
            <div className={clsx('text--center padding-horiz--md', styles.chapterCard)}>
              <h3>Physical AI Fundamentals</h3>
              <p>Understand the core concepts of Physical AI and how it relates to humanoid robotics.</p>
              <Link className="button button--outline button--secondary" to="/docs/modules/ros2-nervous-system/introduction">
                Read Chapter
              </Link>
            </div>
          </div>
          <div className="col col--3">
            <div className={clsx('text--center padding-horiz--md', styles.chapterCard)}>
              <h3>Digital Twin Technologies</h3>
              <p>Learn about simulation environments and digital twin technologies for robotics.</p>
              <Link className="button button--outline button--secondary" to="/docs/modules/digital-twin/introduction">
                Read Chapter
              </Link>
            </div>
          </div>
          <div className="col col--3">
            <div className={clsx('text--center padding-horiz--md', styles.chapterCard)}>
              <h3>AI & Robotics Intelligence</h3>
              <p>Explore AI techniques for robotic control and decision making.</p>
              <Link className="button button--outline button--secondary" to="/docs/modules/ai-robot-brain/introduction">
                Read Chapter
              </Link>
            </div>
          </div>
          <div className="col col--3">
            <div className={clsx('text--center padding-horiz--md', styles.chapterCard)}>
              <h3>Multimodal AI Systems</h3>
              <p>Discover vision-language-action integration for advanced robotics.</p>
              <Link className="button button--outline button--secondary" to="/docs/modules/vla-integration/introduction">
                Read Chapter
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// Textbook Table of Contents component
function TableOfContents() {
  return (
    <section className={clsx(styles.features, styles.tocSection)}>
      <div className="container">
        <h2 className="text--center">Table of Contents</h2>
        <div className="row">
          <div className="col col--6">
            <ul className={styles.tocList}>
              <li><Link to="/docs/modules/ros2-nervous-system/introduction">Chapter 1: Physical AI Fundamentals</Link></li>
              <li><Link to="/docs/modules/ros2-nervous-system/python-agents">Chapter 2: Python Agents for Robotics</Link></li>
              <li><Link to="/docs/modules/ros2-nervous-system/urdf-modeling">Chapter 3: URDF Modeling</Link></li>
              <li><Link to="/docs/modules/digital-twin/introduction">Chapter 4: Digital Twin Technologies</Link></li>
              <li><Link to="/docs/modules/digital-twin/physics-simulation-gazebo">Chapter 5: Physics Simulation in Gazebo</Link></li>
            </ul>
          </div>
          <div className="col col--6">
            <ul className={styles.tocList}>
              <li><Link to="/docs/modules/ai-robot-brain/introduction">Chapter 6: AI & Robotics Intelligence</Link></li>
              <li><Link to="/docs/modules/ai-robot-brain/isaac-sim-synthetic-data">Chapter 7: Isaac Sim & Synthetic Data</Link></li>
              <li><Link to="/docs/modules/ai-robot-brain/isaac-ros-vslam">Chapter 8: Visual SLAM</Link></li>
              <li><Link to="/docs/modules/vla-integration/introduction">Chapter 9: Multimodal AI Systems</Link></li>
              <li><Link to="/docs/modules/vla-integration/cognitive-planning-llms">Chapter 10: Cognitive Planning with LLMs</Link></li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Textbook - ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Textbook - Comprehensive Guide">
      <HomepageHeader />
      <main>
        <ChapterOverview />
        <TableOfContents />
      </main>
    </Layout>
  );
}
