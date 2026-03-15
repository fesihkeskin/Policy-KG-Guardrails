# Policy Description: Healthcare 
*Vesion: v20250308*


This is a sample policy developed by Xu et al. (2015). The policy manages access to electronic health records (HRs) and individual HR items (entries within health records). It defines access rules for nurses, doctors, patients, and authorized agents (such as a patient’s spouse).

**Reference**: Zhongyuan Xu and Scott D. Stoller. *Mining attribute-based access control policies.* IEEE Transactions on Dependable and Secure Computing, 12(5):533–545, September–October 2015.

## Attributes

### Subject/User Attributes
The subjects of this policy include doctros, nurses, patients, and agents within the healthcare facility. The following attributes are used to describe the subjects.

| Attribute Name      | Multiplicity, Type     | Description                                               | Example Values                                     |
|--------------------|------------------|-----------------------------------------------------------|--------------------------------------------------|
| uid          | Single, String   | User’s unique identifier.                                  | carNurse1, oncDoc2, oncPat2, etc.                         |
| position          | Single, String   | The user's position.        | doctor, nurse, patient, agent             |
| specialties        | Multi, Set\<String>   | The user’s areas of medical expertise (for doctors).                     | {cardiology, oncology}, etc.                      |
| teams         | Multi, Set\<String>   | Medical teams of which the user is a member (for doctors).             | {oncTeam1, carTeam2}, etc. |
| ward    | Single, String   | The ward in which the user works (for nurses). | oncWard, carWard, etc.                          |
| agentFor    | Multi, Set\<String>   | The set of patients for which this user is an agent (for agents).                     | {oncPat2, carPat3}, etc.    |


### Resource Attributes
The resources of this policy include electronic health records (HRs) and individual HR items (entries within health records). The following attributes are used to describe the resources.

| Attribute Name      | Multiplicity, Type     | Description                                               | Example Values                                     |
|--------------------|------------------|-----------------------------------------------------------|--------------------------------------------------|
| rid      | Single, String   | Resource’s unique identifier                             | oncPat1nursingItem, oncPat1HR, etc.               |
| type             | Single, String   | Type of resource being accessed.                        | HR, HRitem                       |
| patient       | Single, String   | The patient associated with the HR or HR item.                 | oncPat1, carPat2, etc.                            |
| treatingTeam          | Single, String   | The team treating the associated patient.                      | oncTeam1, carTeam2, etc.                           |
| ward      | Single, String   | The ward in which the associated patient is being treated.            | oncWard, carWard, etc.                             |
| topics      | Multi, Set\<String>   | Medical areas to which the item is relevant (for HR-item resources).            | {cardiology, oncology}, etc.                              |
| author      | Single, String   | ID of the user who created the HR item (for HR-items).            | oncDoc1, carNurse2, etc.                             |


## Rules Set
This section defines the policy rules. SubCond (subject condition) specifies requirements related to the subject’s attributes, while ResCond (resource condition) defines conditions based on resource attributes. cons (constraint) applies conditions that depend on both subject and resource attributes.


### Rules for health records

- Rule 1:  A nurse can add an item in a HR for a patient in the ward in which he/she works.

```rule 1
subCond: position ∈ {nurse}
resCond: type ∈ {HR}
cons: ward = ward
actions: {addItem}
```

- Rule 2: A user can add an item in a HR for a patient treated by one of the teams of which he/she is a member.
```rule 2
subCond: 
resCond: type ∈ {HR}
cons: teams ∋ treatingTeam
actions: {addItem}
```

- Rule 3: A user can add an item with topic "note" in his/her own HR.
```rule 3
subCond: 
resCond: type ∈ {HR}
cons: uid = patient
actions: {addNote}
```

- Rule 4: A user can add an item with topic "note" in the HR of a patient for which he/she is an agent.
```rule 4
subCond: 
resCond: type ∈ {HR}
cons: agentFor ∋ patient
actions: {addNote}
```

### Rules for health record items

- Rule 5: The author of an item can read it.
```rule 5
subCond: 
resCond: type ∈ {HRitem}
cons: uid = author
Actions: {read}
```

- Rule 6: A user can read an item in a HR for a patient treated by one of the teams of which he/she is a member, if the topics of the item are among his/her specialties.
```rule 6
subCond:
resCond: type ∈ {HRitem}
cons: specialties ⊇ topics, teams ∋ treatingTeam
actions: {read}
```


