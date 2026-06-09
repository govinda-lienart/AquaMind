-- MySQL dump 10.13  Distrib 9.3.0, for macos15.2 (arm64)
--
-- Host: localhost    Database: aquamind
-- ------------------------------------------------------
-- Server version	8.4.9

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `annotations`
--

DROP TABLE IF EXISTS `annotations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `annotations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `frame_id` int DEFAULT NULL,
  `class_id` int DEFAULT NULL,
  `label` varchar(255) DEFAULT NULL,
  `x_center` float DEFAULT NULL,
  `y_center` float DEFAULT NULL,
  `width` float DEFAULT NULL,
  `height` float DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `session_id` varchar(30) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_annotations_frame` (`frame_id`),
  CONSTRAINT `annotations_ibfk_1` FOREIGN KEY (`frame_id`) REFERENCES `frames` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5320 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `frames`
--

DROP TABLE IF EXISTS `frames`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `frames` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_id` int DEFAULT NULL,
  `frame_path` varchar(255) DEFAULT NULL,
  `frame_number` int DEFAULT NULL,
  `timestamp` float DEFAULT NULL,
  `extracted_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_video_frame` (`video_id`,`frame_number`),
  CONSTRAINT `fk_frames_video` FOREIGN KEY (`video_id`) REFERENCES `videos` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1662 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `keypoints`
--

DROP TABLE IF EXISTS `keypoints`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `keypoints` (
  `id` int NOT NULL AUTO_INCREMENT,
  `annotation_id` int NOT NULL,
  `name` varchar(50) NOT NULL,
  `x` float NOT NULL,
  `y` float NOT NULL,
  `visible` int NOT NULL DEFAULT '2',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `annotation_id` (`annotation_id`),
  CONSTRAINT `keypoints_ibfk_1` FOREIGN KEY (`annotation_id`) REFERENCES `annotations` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=736 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary view structure for view `session_summary`
--

DROP TABLE IF EXISTS `session_summary`;
/*!50001 DROP VIEW IF EXISTS `session_summary`*/;
SET @saved_cs_client     = @@character_set_client;
/*!50503 SET character_set_client = utf8mb4 */;
/*!50001 CREATE VIEW `session_summary` AS SELECT 
 1 AS `session_id`,
 1 AS `frames`,
 1 AS `total_annotations`,
 1 AS `run_at`,
 1 AS `file_path`*/;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `videos`
--

DROP TABLE IF EXISTS `videos`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `videos` (
  `id` int NOT NULL AUTO_INCREMENT,
  `file_path` varchar(500) DEFAULT NULL,
  `fps` int DEFAULT NULL,
  `resolution` enum('720p','1080p','4K') DEFAULT NULL,
  `session_type` enum('behaviour','tagging','close_up','mixed') DEFAULT NULL,
  `obstacles` tinyint(1) DEFAULT NULL,
  `fish_count` int DEFAULT NULL,
  `notes` text,
  `filmed_at` datetime DEFAULT NULL,
  `species` varchar(100) DEFAULT 'danio_rerio',
  `morph` varchar(100) DEFAULT NULL,
  `tank_width_cm` float DEFAULT NULL,
  `tank_height_cm` float DEFAULT NULL,
  `tank_depth_cm` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `file_path` (`file_path`)
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Final view structure for view `session_summary`
--

/*!50001 DROP VIEW IF EXISTS `session_summary`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_0900_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`%` SQL SECURITY DEFINER */
/*!50001 VIEW `session_summary` AS select `a`.`session_id` AS `session_id`,count(distinct `a`.`frame_id`) AS `frames`,count(0) AS `total_annotations`,min(`a`.`created_at`) AS `run_at`,`v`.`file_path` AS `file_path` from ((`annotations` `a` join `frames` `f` on((`a`.`frame_id` = `f`.`id`))) join `videos` `v` on((`f`.`video_id` = `v`.`id`))) group by `a`.`session_id`,`v`.`file_path` order by `a`.`session_id` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-06-09 10:55:53
