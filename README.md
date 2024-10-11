# Code for *LLM Based Pipeline For Vision Log Retrieval In Automotive Software Engineering*



To use this code, 
  1. Enter your huggingface token in data/hf_token file
  2. Download the *Zenseact Open Dataset (ZOD)* from https://zod.zenseact.com/sequences/.

      Extract the contents of the *vehicle_data.tar.gz* and *images_front_blur.tar.gz* files, rename them to `signal` and `video`, and place them in the `data` folder.
      
      The expected folder structure is:
      
      - `data`
        - `signal`
          - `000000`
          - ...
        - `video`
          - `000000`
          - ...

  3. Use *query.ipynb* file to search for desired scenarios. 
     Refer to  *replication.ipynb* for results mentioned in the paper.
