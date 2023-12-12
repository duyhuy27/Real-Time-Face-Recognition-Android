/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.tranlequyen.facerecognition;

public interface SimilarityClassifier {


  /** Một kết quả không thay đổi được trả về bởi Bộ phân loại mô tả những gì đã được nhận dạng. */
  class Recognition {
    /**
     * Một số nhận dạng duy nhất cho những gì đã được xác thực khuôn mặt. Cụ thể cho lớp, không phải trường hợp của
     *      * đối tượng.
     */
    private final String id;
    /** Hiển thị tên của đối tượng dc xác thực khôn mặt. */
    private final String title;


    private final Float distance;
    private Object extra;

    public Recognition(
            final String id, final String title, final Float distance) {
      this.id = id;
      this.title = title;
      this.distance = distance;
      this.extra = null;

    }

    public void setExtra(Object extra) {
        this.extra = extra;
    }
    public Object getExtra() {
        return this.extra;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (distance != null) {
        resultString += String.format("(%.1f%%) ", distance * 100.0f);
      }

      return resultString.trim();
    }

  }
}
