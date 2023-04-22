import React from "react";

import "./TermsOfService.css";

const TermsOfServicePage = () => {
  return (
    <div className="container rounded border p-4">
      <h1>Terms of Service</h1>
      <div className="">
        Welcome to our Python Notebook Grading Application ("the Application").
        By using the Application, you agree to the following Terms of Service:
        <ol className="list">
          <li className="bulletPoint">
            <i>Acceptance of Terms.</i> These Terms of Service constitute a
            legally binding agreement between you and the Application provider.
            By using the Application, you agree to be bound by these terms, as
            well as any applicable laws and regulations.
          </li>
          <li className="bulletPoint">
            <i>Purpose of the Application.</i> The Application is designed to
            help grade and judge the readability of Python notebooks. It is
            intended for research purposes and for subjective grading in
            development teams. It is not intended to provide any guarantees or
            assurances regarding the quality, accuracy, or completeness of any
            Python notebook.
          </li>
          <li className="bulletPoint">
            <i>User Conduct.</i> By using the Application, you agree to use it
            only for lawful purposes and in a manner that does not infringe upon
            the rights of others. You agree not to use the Application to
            distribute any unlawful, harmful, or offensive material, or to
            engage in any conduct that could damage, disable, or impair the
            Application.
          </li>
          <li className="bulletPoint">
            <i>Ownership and Intellectual Property.</i> The Application and all
            of its contents, including but not limited to text, graphics,
            images, and software, are owned by the Application provider and are
            protected by copyright and other intellectual property laws. You
            agree not to copy, modify, distribute, or otherwise use any part of
            the Application without the express written consent of the
            Application provider.
          </li>
          <li className="bulletPoint">
            <i>Disclaimer of Liability.</i> The Application provider assumes no
            liability for any errors, omissions, or inaccuracies in the
            Application or any Python notebook submitted through the
            Application. The Application is provided "as is" and without
            warranties of any kind, express or implied, including but not
            limited to warranties of merchantability, fitness for a particular
            purpose, and non-infringement.
          </li>
          <li className="bulletPoint">
            <i>Limitation of Liability.</i> In no event shall the Application
            provider be liable for any damages, including but not limited to
            direct, indirect, incidental, special, or consequential damages,
            arising out of or in connection with the use or inability to use the
            Application or any Python notebook submitted through the
            Application.
          </li>
          <li className="bulletPoint">
            <i>Indemnification.</i> You agree to indemnify and hold harmless the
            Application provider and its officers, directors, employees, and
            agents from any and all claims, liabilities, damages, and expenses
            (including attorneys' fees) arising out of or in connection with
            your use of the Application or any Python notebook submitted through
            the Application.
          </li>
          <li className="bulletPoint">
            <i>Termination.</i> The Application provider reserves the right to
            terminate your use of the Application at any time and for any
            reason, without notice or liability.
          </li>
          <li className="bulletPoint">
            <i>Governing Law.</i> These Terms of Service shall be governed by
            and construed in accordance with the laws of the jurisdiction in
            which the Application provider is located, without regard to its
            conflict of laws principles.
          </li>
          <li className="bulletPoint">
            <i>Entire Agreement.</i> These Terms of Service constitute the
            entire agreement between you and the Application provider regarding
            the use of the Application, and supersede all prior agreements and
            understandings, whether written or oral.
          </li>
        </ol>
      </div>
    </div>
  );
};

export default TermsOfServicePage;
