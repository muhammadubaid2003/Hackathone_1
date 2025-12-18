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

// What this textbook covers component
function TextbookCoverage() {
  return (
    <section className={clsx(styles.features, styles.coverageSection)}>
      <div className="container">
        <h2>What this Textbook Covers</h2>
        <div className="row">
          <div className="col col--6">
            <h3>Core Concepts & Technologies</h3>
            <ul className={styles.coverageList}>
              <li>Physical AI Fundamentals and ROS 2 architecture</li>
              <li>Python agents for robotics applications</li>
              <li>URDF modeling for robot design</li>
              <li>Digital twin technologies with Gazebo and Unity</li>
              <li>Physics simulation and sensor modeling</li>
            </ul>
          </div>
          <div className="col col--6">
            <h3>Advanced AI & Robotics</h3>
            <ul className={styles.coverageList}>
              <li>AI-powered robot brain with NVIDIA Isaac</li>
              <li>Synthetic data generation techniques</li>
              <li>Visual SLAM for navigation</li>
              <li>Humanoid navigation systems</li>
              <li>Vision-Language-Action integration</li>
            </ul>
          </div>
        </div>
        <div className="row" style={{marginTop: '2rem'}}>
          <div className="col col--12">
            <h3>Practical Applications</h3>
            <div className={styles.applicationBox}>
              <p>This textbook provides hands-on guidance for building intelligent humanoid robots, from fundamental concepts to advanced multimodal AI systems. You'll learn to integrate cutting-edge technologies for real-world robotics applications.</p>
            </div>
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
        <TextbookCoverage />
      </main>
    </Layout>
  );
}
