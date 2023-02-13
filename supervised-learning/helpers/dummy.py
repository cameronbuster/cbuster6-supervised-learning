#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import print_function

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class DummiesCreate(BaseEstimator, TransformerMixin):
    """
    This class creates dummy variables in a dataset. The fit method creates the dictvectorizer object, and the
    transform method uses it to transform the dataset according to the initial data.
    Drops columns related to missing categorical features (i.e. categorical_feature=nan)
    """

    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def transform(self, df, **transform_params):
        df_dummied = pd.DataFrame(self.v.transform(row for _, row in df.iterrows()), columns=self.v.feature_names_)
        # df_dummied = df_dummied.drop([col for col in df_dummied.columns if col.endswith("=nan")], axis=1) #makes models worse
        return df_dummied

    def fit(self, df, y=None, **fit_params):
        self.v.fit(row for _, row in df.iterrows())
        return self
