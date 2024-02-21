# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import unittest


from cornac.datasets import mind as mind
import numpy as np


class TestMind(unittest.TestCase):

    def test_mind(self):
        ratings = mind.load_feedback(
            fpath="./tests/enriched_data/dummy_data/mind_uir.csv")
        expected = 2690
        actual = len(ratings)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_load_features_json(self):
        sentiment = mind.load_sentiment(
            fpath="./tests/enriched_data/sentiment.json")
        self.assertTrue(abs(0.6597 - sentiment['N55189']) < 0.001)
        self.assertTrue(abs(-0.9842 - sentiment['N17228']) < 0.001)
        self.assertTrue(abs(-0.9914 - sentiment['N38001']) < 0.001)
        self.assertTrue(42386 - len(sentiment) < 0.001)
        category = mind.load_category(
            fpath="./tests/enriched_data/category.json")
        self.assertTrue("tv" == category['N55189'])
        self.assertTrue("news" == category['N46039'])
        self.assertTrue("news" == category['N42078'])
        self.assertTrue(42386 - len(category) < 0.001)
        complexity = mind.load_complexity(
            fpath="./tests/enriched_data/complexity.json")
        self.assertTrue(abs(29.1167938931 - complexity['N55189']) < 0.001)
        self.assertTrue(abs(14.9315415822 - complexity['N46039']) < 0.001)
        self.assertTrue(abs(8.4084782609 - complexity['N42078']) < 0.001)
        self.assertTrue(36580 - len(complexity) < 0.001)

        story = mind.load_story(fpath="./tests/enriched_data/story.json")
        self.assertTrue(abs(458 - story['N55189']) < 0.001)
        self.assertTrue(abs(0 - story['N46039']) < 0.001)
        self.assertTrue(abs(0 - story['N42078']) < 0.001)
        self.assertTrue(42386 - len(story) < 0.001)

        entities = mind.load_entities(
            fpath="./tests/enriched_data/party.json")
        self.assertCountEqual(entities['N51741'], ["Democratic Party"])
        self.assertCountEqual(entities['N43369'], [
                              "Arab Socialist Union", "National Democratic Party"])
        self.assertCountEqual(entities['N49550'], ["Republican Party", "Republican Party", "Republican Party", "Republican Party", "Republican Party",
                                                   "Republican Party", "Republican Party", "Socialist Workers Party", "Socialist Workers Party", "Socialist Workers Party",
                                                   "Socialist Workers Party", "Socialist Workers Party", "Socialist Workers Party", "Socialist Workers Party"])
        self.assertTrue(5060 - len(entities) < 0.001)

        genre = mind.load_category_multi(
            fpath="./tests/enriched_data/dummy_data/enhanced_category.json")
        # print(genre)
        self.assertTrue(5 - genre['N55189'].shape[0] < 0.001)
        self.assertEqual(genre['N55189'].tolist(),
                         np.array([1, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre['N45794'].tolist(),
                         np.array([0, 0, 1, 0, 0]).tolist())

        genre1 = mind.load_category_multi(
            fpath="./tests/enriched_data/category.json")
        self.assertTrue(17 - genre1['N55189'].shape[0] < 0.001)
        self.assertEqual(genre1['N55189'].tolist(),
                         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre1['N47686'].tolist(),
                         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre1['N62365'].tolist(),
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre1['N11713'].tolist(),
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).tolist())
        self.assertEqual(genre1['N41467'].tolist(),
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).tolist())
        min_maj = mind.load_min_maj(
            fpath="./tests/enriched_data/min_maj.json")
        self.assertTrue(2 - min_maj['N55189'].shape[0] < 0.001)
        self.assertTrue(0.9412 - min_maj['N55189'][0] < 0.001)
        self.assertTrue(0.0588 - min_maj['N55189'][1] < 0.001)
        self.assertTrue(0.9412 - min_maj['N55189'][0] < 0.001)
        self.assertTrue('N12449' not in min_maj)

    def test_load_multi_category(self):
        genre1 = mind.load_category_multi(
            fpath="./tests/enriched_data/dummy_data/category_multi.json")
        self.assertEqual(genre1['N55528'].tolist(),
                         np.array([1, 1, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre1['N18955'].tolist(),
                         np.array([0, 1, 1, 0, 0, 0]).tolist())
        self.assertEqual(genre1['N61837'].tolist(),
                         np.array([0, 0, 0, 1, 1, 0]).tolist())
        self.assertEqual(genre1['N53526'].tolist(),
                         np.array([0, 1, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre1['N38324'].tolist(),
                         np.array([0, 1, 0, 0, 0, 1]).tolist())
        self.assertEqual(genre1['N2073'].tolist(),
                         np.array([0, 0, 1, 0, 0, 0]).tolist())

    def test_load_feature_csv(self):
        sentiment = mind.load_sentiment(
            fpath="./tests/enriched_data/dummy_data/sentiment.csv")
        self.assertTrue(abs(0.0389 - sentiment['N18438']) < 0.001)
        self.assertTrue(abs(0.0308 - sentiment['N17228']) < 0.001)
        self.assertTrue(abs(0.25988 - sentiment['N37918']) < 0.001)
        self.assertTrue(42386 - len(sentiment) < 0.001)
        category = mind.load_category(
            fpath="./tests/enriched_data/dummy_data/category.csv")
        self.assertTrue("tv" == category['N55189'])
        self.assertTrue("news" == category['N46039'])
        self.assertTrue("news" == category['N42078'])
        self.assertTrue(42386 - len(category) < 0.001)
        complexity = mind.load_complexity(
            fpath="./tests/enriched_data/dummy_data/complexity.csv")
        self.assertTrue(abs(76.35 - complexity['N61720']) < 0.001)
        self.assertTrue(abs(59.13 - complexity['N31891']) < 0.001)
        self.assertTrue(abs(73.37 - complexity['N40107']) < 0.001)
        self.assertTrue(42386 - len(complexity) < 0.001)

        story = mind.load_story(
            fpath="./tests/enriched_data/dummy_data/story.csv")
        self.assertTrue(abs(533 - story['N13111']) < 0.001)
        self.assertTrue(abs(0 - story['N1512']) < 0.001)
        self.assertTrue(abs(1090 - story['N17101']) < 0.001)
        self.assertTrue(42386 - len(story) < 0.001)

        entities = mind.load_entities(
            fpath="./tests/enriched_data/dummy_data/entities_example.csv")
        self.assertCountEqual(entities['N6892'], [
                              "Democratic Party", "Democratic Party"])
        self.assertCountEqual(entities['N30549'], [
                              "Republican Party", "Republican Party", "Democratic Party", "Democratic Party"])
        self.assertTrue(3 - len(entities) < 0.001)

        genre = mind.load_category_multi(
            fpath="./tests/enriched_data/dummy_data/category_multi_example.csv")
        self.assertTrue(6 - genre['N55528'].shape[0] < 0.001)
        self.assertEqual(genre['N55528'].tolist(),
                         np.array([1, 1, 0, 0, 0, 0]).tolist())
        self.assertEqual(genre['N53526'].tolist(),
                         np.array([0, 0, 0, 1, 0, 0]).tolist())

        min_maj = mind.load_min_maj(
            fpath="./tests/enriched_data/dummy_data/min_maj_example.csv")
        self.assertTrue(2 - min_maj['N55528'].shape[0] < 0.001)
        self.assertTrue(0.2 - min_maj['N2073'][0] < 0.001)
        self.assertTrue(0.8 - min_maj['N2073'][1] < 0.001)
        self.assertTrue(0.1 - min_maj['N53526'][0] < 0.001)


if __name__ == '__main__':
    unittest.main()
